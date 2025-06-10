# extract_query.py (finalized for SHL RAG)

import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import time
import re
from langchain_huggingface import HuggingFaceEndpoint
import os

from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_ollama import OllamaLLM

class JobCoreDetails(BaseModel):
    key_responsibilities: list[str] = Field(description="A list of the 3-5 most important responsibilities or tasks.")
    required_skills: list[str] = Field(description="A list of essential technical and soft skills mentioned.")
    education_requirements: list[str] = Field(description="Minimum educational qualifications required.")

def extract_linkedin_job_description(url: str) -> str:
    print(f"Attempting specialized extraction from LinkedIn URL: {url}...")
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
    chrome_options.add_argument("--window-size=1920,1080") # Add this for consistent rendering

    driver = webdriver.Chrome(options=chrome_options)
    
    try:
        driver.get(url)
        time.sleep(3) # Increased initial sleep

        wait = WebDriverWait(driver, 25) # Increased timeout to 25 seconds

        # Try to wait for one of the common job description containers
        try:
            print("Waiting for 'jobs-description__content' to be present...")
            wait.until(EC.presence_of_element_located((By.CLASS_NAME, "jobs-description__content")))
            print("'jobs-description__content' found.")
        except TimeoutException:
            print("Timeout waiting for 'jobs-description__content'. Trying 'description__text' directly...")
            try:
                wait.until(EC.presence_of_element_located((By.CLASS_NAME, "description__text")))
                print("'description__text' found directly.")
            except TimeoutException:
                print("Timeout waiting for 'description__text' directly. Job description likely not loaded.")
                return "Job description not found."

        # Attempt to click 'Show more' button if it exists
        try:
            show_more_button = WebDriverWait(driver, 5).until( # Shorter wait for button
                EC.element_to_be_clickable((By.CLASS_NAME, "show-more-less-html__button"))
            )
            driver.execute_script("arguments[0].click();", show_more_button)
            print("Clicked the 'Show more' button.")
            time.sleep(1) 
        except TimeoutException:
            print("No 'Show more' button found within 5 seconds or not clickable.")
        except NoSuchElementException:
            print("No 'Show more' button found.")
        except Exception as e:
            print(f"Could not click 'Show more' button due to unexpected error: {e}")

        soup = BeautifulSoup(driver.page_source, 'html.parser')
        
        description_container = soup.find('div', class_='description__text') 
        if not description_container: 
            description_container = soup.find('div', class_='jobs-description__content')

        if description_container:
            job_description = description_container.get_text(separator='\n', strip=True)
            print("Successfully extracted the complete job description from LinkedIn.")
            return job_description
        else:
            print("Could not find the job description container or its text within the LinkedIn page after all attempts.")
            return "Job description not found."

    except TimeoutException as e:
        print(f"Global TimeoutError: {e}. LinkedIn page content was not found within specified time.")
        return "Job description not found."
    except Exception as e:
        print(f"[LinkedIn Extraction Critical Error]: {e}")
        return "Job description not found."
    finally:
        if 'driver' in locals():
            driver.quit()

def analyze_job_description_for_core_details(text: str) -> JobCoreDetails | None:
    if not text or text == "Job description not found.":
        return None

    parser = PydanticOutputParser(pydantic_object=JobCoreDetails)
    prompt_template = """
    You are an expert HR analyst. Your task is to analyze the following job description text and extract only the key responsibilities, required skills, and educational qualifications in a structured JSON format.
    Be as expert as possible, focusing on the most relevant details for a hiring manager. Look most carefully for the key responsibilities, required skills, and educational qualifications.
    Do not include any other information or context.
    
    {format_instructions}
    
    Here is the job description:
    ---
    {job_text}
    ---
    """
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["job_text"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    # Try Ollama first
    try:
        llm = OllamaLLM(model="phi3:mini", temperature=0)
        chain = prompt | llm | parser
        result = chain.invoke({"job_text": text})
        return result
    except Exception as ollama_exc:
        print(f"Ollama failed or not running: {ollama_exc}. Falling back to Hugging Face API...")
        try:
            os.environ["HUGGINGFACEHUB_API_TOKEN"] = "REMOVED_HF_TOKEN"
            llm = HuggingFaceEndpoint(
                repo_id="HuggingFaceH4/zephyr-7b-beta",
                temperature=0.01,  # Must be strictly positive for HF API
                max_new_tokens=512
            )
            chain = prompt | llm | parser
            result = chain.invoke({"job_text": text})
            return result
        except Exception as hf_exc:
            print(f"Hugging Face API also failed: {hf_exc}")
            return None

def extract_query_text(user_query: str) -> str:
    final_query_for_recommender = ""
    urls_found = re.findall(r'https?://\S+|www\.\S+', user_query)
    if urls_found:
        # If a URL is present, extract and process
        extracted_url = urls_found[0]
        recommendation_phrase_match = re.search(r'(can you recommend|looking for an assessment)[^.]*\.?\s*Time limit is less than \d+ minutes\.', user_query, re.IGNORECASE)
        original_accompanying_text = ""
        if recommendation_phrase_match:
            original_accompanying_text = recommendation_phrase_match.group(0).strip()
        raw_jd_from_url = extract_linkedin_job_description(extracted_url)
        if raw_jd_from_url and raw_jd_from_url != "Job description not found.":
            core_jd_details = analyze_job_description_for_core_details(raw_jd_from_url)
            if core_jd_details:
                responsibilities_str = "; ".join(core_jd_details.key_responsibilities) or "N/A"
                skills_str = "; ".join(core_jd_details.required_skills) or "N/A"
                education_str = "; ".join(core_jd_details.education_requirements) or "N/A"
                leftover_text = user_query.replace(extracted_url, "").replace(original_accompanying_text, "").strip()
                if original_accompanying_text:
                    final_query_for_recommender = f"Here is requirement of Responsibilities: {responsibilities_str}, with skill set {skills_str} and qualification {education_str}. {original_accompanying_text}"
                else:
                    final_query_for_recommender = (
                        f"I am looking for assessment(s) for a role with Responsibilities: {responsibilities_str}, "
                        f"Required Skills: {skills_str}, and Education: {education_str}. "
                        "Please recommend relevant assessments."
                    )
                if leftover_text:
                    final_query_for_recommender += f" {leftover_text}"
            else:
                final_query_for_recommender = user_query.replace(extracted_url, "").strip() or "No relevant text found."
        else:
            final_query_for_recommender = user_query.replace(extracted_url, "").strip() or "No relevant text found."
    else:
        # If no URL, just return the plain query as is
        final_query_for_recommender = user_query.strip()
    return final_query_for_recommender


# ans=extract_query_text("Here is a JD text https://www.linkedin.com/jobs/view/4221193932/?alternateChannel=search&refId=%2BiYLwYzJi47r2mte%2BrMCwQ%3D%3D&trackingId=2ieo%2BTuou2NBotCLIWDd3g%3D%3D , can you recommend some assessment that can help me screen applications. Time limit is less than 30 minutes")
# print(ans)