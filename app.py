import os
import io
import base64
import json
import re
import fitz
import pandas as pd
import streamlit as st
from PIL import Image
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configure Google Generative AI
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

class HRAnalyticsSystem:
    def __init__(self):
        self.model = genai.GenerativeModel('gemini-1.5-flash')

    def convert_pdf_to_images(self, file_bytes):
        """Convert uploaded PDF file to images (first page only)."""
        pdf_document = fitz.open(stream=file_bytes, filetype="pdf")
        images = []
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            pix = page.get_pixmap()
            img_byte_arr = io.BytesIO(pix.tobytes())
            images.append(Image.open(img_byte_arr))
        return images

    def input_pdf_setup(self, uploaded_file):
        """Prepare PDF file for AI processing."""
        if uploaded_file is not None:
            images = self.convert_pdf_to_images(uploaded_file.read())
            first_page = images[0]
            img_byte_arr = io.BytesIO()
            first_page.save(img_byte_arr, format='JPEG')
            img_byte_arr = img_byte_arr.getvalue()
            return [{
                "mime_type": "image/jpeg",
                "data": base64.b64encode(img_byte_arr).decode()
            }]
        else:
            raise FileNotFoundError("No file uploaded")

    def get_gemini_response(self, input_text, pdf_content, prompt):
        """Get AI-generated response from Google Gemini model."""
        response = self.model.generate_content([input_text, pdf_content[0], prompt])
        return response.text if hasattr(response, 'text') else "No valid response received."

    def format_json_response(self, text):
        """Ensure AI response is formatted as JSON."""
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            try:
                json_match = re.search(r'\{[\s\S]*\}', text)
                if json_match:
                    return json.loads(json_match.group(0))
                else:
                    return {
                        "analysis": text,
                        "status": "Formatted as text",
                        "timestamp": str(pd.Timestamp.now())
                    }
            except Exception as e:
                return {"error": "Could not parse response", "raw_text": text, "status": str(e)}

    def analyze_sentiment(self, feedback_text):
        """Perform sentiment analysis on employee feedback."""
        prompt = """
        You are an AI HR analyst. Analyze the following employee feedback and return ONLY a JSON object:
        {
            "sentiment": { "overall": "POSITIVE/NEGATIVE/NEUTRAL", "score": "0-100" },
            "key_concerns": ["concern1", "concern2"],
            "attrition_risk": { "level": "HIGH/MEDIUM/LOW", "factors": ["factor1", "factor2"] },
            "recommendations": ["recommendation1", "recommendation2"],
            "priority_areas": ["area1", "area2"]
        }
        """
        try:
            response = self.model.generate_content([prompt, feedback_text])
            return self.format_json_response(response.text)
        except Exception as e:
            return {"error": "Analysis failed", "message": str(e), "status": "Error"}

# Streamlit UI
def main():
    st.set_page_config(page_title="HR Analytics Suite", layout="wide")

    hr_system = HRAnalyticsSystem()

    tabs = st.tabs(["Resume Screening", "Employee Sentiment", "Analytics Dashboard"])

    # Resume Screening Tab
    with tabs[0]:
        st.header("ATS Resume Screening")
        input_text = st.text_area("Job Description:", key="input")
        uploaded_file = st.file_uploader("Upload your resume (PDF)...", type=["pdf"])

        if uploaded_file is not None:
            st.write("PDF Uploaded Successfully")

        input_prompt1 = "Analyze the resume and provide insights based on the job description."
        input_prompt2 = "Suggest skills improvements based on the job description."
        input_prompt3 = "Provide a percentage match between the resume and job description."

        col1, col2, col3 = st.columns(3)
        with col1:
            submit1 = st.button("Tell me about the Resume")
        with col2:
            submit2 = st.button("How can I improve my skills")
        with col3:
            submit3 = st.button("Percentage match")

        if uploaded_file is not None:
            pdf_content = hr_system.input_pdf_setup(uploaded_file)

            if submit1:
                responses = hr_system.get_gemini_response(input_text, pdf_content, input_prompt1)
                st.subheader("Analysis Report")
                st.write(responses)

            elif submit2:
                responses = hr_system.get_gemini_response(input_text, pdf_content, input_prompt2)
                st.subheader("Skill Improvement Suggestions")
                st.write(responses)

            elif submit3:
                responses = hr_system.get_gemini_response(input_text, pdf_content, input_prompt3)
                st.subheader("Percentage Match")
                st.write(responses)

    # Employee Sentiment Analysis Tab
    with tabs[1]:
        st.header("Employee Sentiment Analysis")
        feedback = st.text_area("Employee Feedback:", height=150, key="feedback")

        if st.button("Analyze Feedback"):
            if feedback:
                with st.spinner("Analyzing feedback..."):
                    analysis = hr_system.analyze_sentiment(feedback)

                    if "error" not in analysis:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.subheader("Sentiment Analysis")
                            st.write(f"Overall: {analysis['sentiment']['overall']}")
                            st.write(f"Score: {analysis['sentiment']['score']}")
                            st.subheader("Attrition Risk")
                            st.write(f"Level: {analysis['attrition_risk']['level']}")
                            st.write("Risk Factors:")
                            for factor in analysis['attrition_risk']['factors']:
                                st.write(f"- {factor}")
                        with col2:
                            st.subheader("Key Concerns")
                            for concern in analysis["key_concerns"]:
                                st.write(f"- {concern}")
                            st.subheader("Recommendations")
                            for rec in analysis["recommendations"]:
                                st.write(f"- {rec}")
                    else:
                        st.error(f"Error in analysis: {analysis['message']}")
            else:
                st.warning("Please provide employee feedback to analyze")

    # Analytics Dashboard Tab
    with tabs[2]:
        st.header("Analytics Dashboard")
        st.info("Analytics dashboard features coming soon...")

if __name__ == "__main__":
    main()
