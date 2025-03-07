%%writefile app.py
import streamlit as st
import pydicom
import numpy as np
import cv2
import tempfile

# Import your specialized modules
from phi.agent import Agent
from phi.model.google import Gemini
from phi.tools.duckduckgo import DuckDuckGo
from langchain_google_genai import ChatGoogleGenerativeAI

# Define the analysis query for the medical agent
analysis_query = """
You are a highly skilled medical imaging expert with extensive knowledge in radiology and diagnostic imaging. Analyze the patient's medical image and structure your response as follows:

### 1. Image Type & Region
- Specify imaging modality (X-ray/MRI/CT/Ultrasound/etc.)
- Identify the patient's anatomical region and positioning
- Comment on image quality and technical adequacy

### 2. Key Findings
- List primary observations systematically
- Note any abnormalities in the patient's imaging with precise descriptions
- Include measurements and densities where relevant
- Describe location, size, shape, and characteristics
- Rate severity: Normal/Mild/Moderate/Severe

### 3. Diagnostic Assessment
- Provide primary diagnosis with confidence level
- List differential diagnoses in order of likelihood
- Support each diagnosis with observed evidence from the patient's imaging
- Note any critical or urgent findings

### 4. Patient-Friendly Explanation
- Explain the findings in simple, clear language that the patient can understand
- Avoid medical jargon or provide clear definitions
- Include visual analogies if helpful
- Address common patient concerns related to these findings

### 5. Research Context
IMPORTANT: Use the DuckDuckGo search tool to:
- Find recent medical literature about similar cases
- Search for standard treatment protocols
- Provide a list of relevant medical links as well
- Research any relevant technological advances
- Include 2-3 key references to support your analysis

Format your response using clear markdown headers and bullet points. Be concise yet thorough.
"""

def main():
    st.title("🩺 Medical Image Analysis")
    st.write("Yo! Upload your DICOM file via the sidebar and select the slice you want to analyze using the buttons below the image.")

    # Upload functionality in the sidebar
    uploaded_file = st.sidebar.file_uploader("Upload a DICOM file", type=["dcm", "DCM","tif","dicom"])
    
    if uploaded_file is not None:
        try:
            # Read the DICOM file directly from the uploaded file
            dicom_data = pydicom.dcmread(uploaded_file)
            st.success("DICOM file uploaded successfully!")
            
            if 'PixelData' in dicom_data:
                image_data = dicom_data.pixel_array
                total=image_data.shape[0]
                # For 3D images, initialize session state for slice index
                if len(image_data.shape) == 3:
                    if "slice_idx" not in st.session_state:
                        st.session_state.slice_idx = 0

                    # Display the currently selected slice
                    selected_slice = image_data[st.session_state.slice_idx]
                    
                    # Normalize the image for display if necessary
                    if selected_slice.dtype != "uint8":
                        selected_slice = cv2.normalize(selected_slice, None, 0, 255, cv2.NORM_MINMAX)
                        selected_slice = np.uint8(selected_slice)
                    
                    st.image(selected_slice, caption=f"Slice {st.session_state.slice_idx + 1}  of Total {total}", use_column_width=True)
                    
                    # Create Previous and Next buttons below the image
                    # Create a row with three equally sized columns
                    # Create three columns with equal width
                    col_prev, col_analyze, col_next = st.columns([2.9, 3, 1])

                    # Define the buttons within their respective columns
                    with col_prev:
                        prev_clicked = st.button("Previous")
                    with col_analyze:
                        analyze_clicked = st.button("Analyze")
                    with col_next:
                        next_clicked = st.button("Next")
                    # Handle the button clicks outside the column context
                    if prev_clicked:
                        if st.session_state.slice_idx > 0:
                            st.session_state.slice_idx -= 1
                        st.rerun()

                    if next_clicked:
                        if st.session_state.slice_idx < image_data.shape[0] - 1:
                            st.session_state.slice_idx += 1
                        st.rerun()

                    if analyze_clicked:
                            with st.spinner("Analyzing..."):
                                # Save the selected slice to a temporary file
                               with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
                                temp_image_path = temp_file.name

                                min_val = np.min(selected_slice)
                                max_val = np.max(selected_slice)

                                normalized_image = ((selected_slice - min_val) / (max_val - min_val) * 255).astype(np.uint8)


                                cv2.imwrite(temp_image_path, normalized_image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

                                
                                # Initialize the medical agent (replace the API key with your own)
                                medical_agent = Agent(
                                    model=Gemini(
                                        api_key="AIzaSyBe5hCcwzCBrR1yeMMxh5ElHhvYPaqbLTQ",  # <-- Replace with your Gemini API key
                                        id="gemini-2.0-flash"
                                    ),
                                    tools=[DuckDuckGo()],
                                    markdown=True
                                )
                                
                                # Run analysis on the selected slice image
                                response = medical_agent.run(analysis_query, images=[temp_image_path])
                                st.markdown("### Analysis for Selected Slice")
                                st.write(response.content)
                                
                                # Optionally, generate an overall summary using the LLM
                                llm = ChatGoogleGenerativeAI(
                                    model="gemini-2.0-flash",
                                    api_key="AIzaSyBe5hCcwzCBrR1yeMMxh5ElHhvYPaqbLTQ",  # <-- Replace with your Gemini API key
                                    temperature=0.0
                                )
                                summary_prompt = f"""
                                Here is the analysis of the selected slice:
                                {response.content}

                                Please provide a complete summary and overall diagnosis based on the analysis.
                                """
                                summary_response = llm.invoke(summary_prompt)
                                st.markdown("### Overall Summary & Diagnosis")
                                st.write(summary_response.content)
                        
                else:
                    st.write("The uploaded DICOM file contains 2D image data.")
                    normalized_image = cv2.normalize(image_data, None, 0, 255, cv2.NORM_MINMAX)
                    # Apply histogram equalization for better contrast
                    #equalized_image = cv2.equalizeHist(image_data.astype(np.uint8))
                    st.image(normalized_image, caption="DICOM Image", use_column_width=True)
                    if st.button("Analyze"):
                      with st.spinner("Analyzing..."):
                          # Save the 2D image to a temp file for passing to the LLM
                          # Save the 2D image to a temp file for passing to the LLM
                          with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
                              temp_image_path = temp_file.name
                              # Convert to 8-bit if necessary (DICOM images can be 16-bit or 12-bit)
                              normalized_image = cv2.normalize(image_data, None, 0, 255, cv2.NORM_MINMAX)
                              cv2.imwrite(temp_image_path, normalized_image.astype('uint8'), [int(cv2.IMWRITE_JPEG_QUALITY), 90])


                          # Initialize medical agent
                          medical_agent = Agent(
                              model=Gemini(
                                  api_key="AIzaSyBe5hCcwzCBrR1yeMMxh5ElHhvYPaqbLTQ",  # Replace with your own Gemini API key
                                  id="gemini-2.0-flash"
                              ),
                              tools=[DuckDuckGo()],
                              markdown=True
                          )

                          # Run analysis with image
                          response = medical_agent.run(analysis_query, images=[temp_image_path])
                          st.markdown("### Analysis Results")
                          st.write(response.content)

                          # Optional: Generate overall summary and diagnosis
                          llm = ChatGoogleGenerativeAI(
                              model="gemini-2.0-flash",
                              api_key="AIzaSyBe5hCcwzCBrR1yeMMxh5ElHhvYPaqbLTQ",  # Replace with your own Gemini API key
                              temperature=0.0
                          )
                          summary_prompt = f"""
                          Here is the analysis of the medical image:
                          {response.content}

                          Please provide a complete summary and overall diagnosis based on the analysis.
                          """

                          summary_response = llm.invoke(summary_prompt)
                          st.markdown("### Overall Summary & Diagnosis")
                          st.write(summary_response.content)
            else:
                st.error("No pixel data found in the DICOM file.")
        except Exception as e:
            st.error(f"Error processing DICOM file: {e}")

if __name__ == "__main__":
    main()

