import streamlit as st
from PIL import Image
import google.generativeai as genai
from gemini_helper import GeminiInspector

# PDF generation imports
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.colors import black, blue, gray
from reportlab.lib.units import inch
import io

def resize_image_for_pdf(image, max_width=4*inch, max_height=5*inch):
    """
    Resize an image to fit within specified maximum dimensions while maintaining aspect ratio.
    
    Args:
        image (PIL.Image): Input image
        max_width (float): Maximum width in PDF units (inches)
        max_height (float): Maximum height in PDF units (inches)
    
    Returns:
        PIL.Image: Resized image
    """
    # Convert image to RGB if it's not already
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Get original image dimensions
    orig_width, orig_height = image.size
    
    # Calculate aspect ratio
    aspect_ratio = orig_width / orig_height
    
    # Calculate new dimensions
    if orig_width > max_width or orig_height > max_height:
        # First check width constraint
        if orig_width > max_width:
            new_width = max_width
            new_height = new_width / aspect_ratio
            
            # If new height exceeds max height, adjust based on height
            if new_height > max_height:
                new_height = max_height
                new_width = new_height * aspect_ratio
        
        # Then check height constraint
        elif orig_height > max_height:
            new_height = max_height
            new_width = new_height * aspect_ratio
        
        # Resize the image
        image = image.resize((int(new_width), int(new_height)), Image.LANCZOS)
    
    return image

def generate_pdf_report(messages, current_image=None):
    """
    Generate a PDF report with clear, professional formatting.
    
    Args:
        messages (list): List of chat messages
        current_image (PIL.Image, optional): Image to include in the report
    
    Returns:
        io.BytesIO: PDF report buffer
    """
    # Create PDF buffer
    pdf_buffer = io.BytesIO()
    
    # Get sample styles and create custom styles
    styles = getSampleStyleSheet()
    title_style = styles['Title'].clone('ReportTitle')
    title_style.fontName = 'Helvetica-Bold'
    title_style.fontSize = 16
    title_style.textColor = blue
    
    section_header_style = styles['Heading2'].clone('SectionHeader')
    section_header_style.fontName = 'Helvetica-Bold'
    section_header_style.textColor = black
    
    user_msg_style = ParagraphStyle(
        'UserMessageStyle',
        parent=styles['BodyText'],
        fontName='Helvetica-Bold',
        fontSize=11,
        textColor=blue,
        spaceBefore=12,
        spaceAfter=6
    )
    
    assistant_msg_style = ParagraphStyle(
        'AssistantMessageStyle',
        parent=styles['BodyText'],
        fontName='Helvetica',
        fontSize=11,
        textColor=black,
        spaceBefore=12,
        spaceAfter=6
    )
    
    # Create PDF document
    doc = SimpleDocTemplate(
        pdf_buffer, 
        pagesize=letter, 
        rightMargin=54, 
        leftMargin=54, 
        topMargin=54, 
        bottomMargin=36
    )
    
    # Prepare story (PDF content)
    story = []
    
    # Add title
    story.append(Paragraph("Construction Invoice/Estimate Analysis Report", title_style))
    story.append(Spacer(1, 18))
    
    # Optional: Add analyzed image to the report
    if current_image:
        # Resize image to fit page width
        resized_image = resize_image_for_pdf(current_image)
        
        img_path = "temp_invoice_image.png"
        resized_image.save(img_path)
        
        img = RLImage(img_path, width=None, height=None)
        img.hAlign = 'CENTER'
        story.append(img)
        story.append(Spacer(1, 18))
    
    # Add chat message sections
    for msg in messages:
        # Add section header
        story.append(Paragraph(msg['role'].capitalize(), section_header_style))
        
        # Add message content
        msg_style = user_msg_style if msg['role'] == 'user' else assistant_msg_style
        story.append(Paragraph(msg['content'], msg_style))
        story.append(Spacer(1, 12))
    
    # Build PDF
    doc.build(story)
    
    # Reset buffer position
    pdf_buffer.seek(0)
    
    return pdf_buffer

# Page configuration
st.set_page_config(
    page_title="Construction Invoice/Estimate Analyzer",
    page_icon="📊",
    layout="wide"
)

# Initialize session state
if 'chat' not in st.session_state:
    st.session_state.chat = None
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'image_analyzed' not in st.session_state:
    st.session_state.image_analyzed = False
if 'current_image' not in st.session_state:
    st.session_state.current_image = None

# Sidebar for API key
with st.sidebar:
    st.title("Settings")
    api_key = st.text_input("Enter Gemini API Key", type="password")
    if api_key:
        inspector = GeminiInspector(api_key)
    else:
        inspector = GeminiInspector()
    
    # Add clear chat button
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.session_state.chat = inspector.start_chat()
        st.session_state.image_analyzed = False
        st.rerun()

# Main title
st.title("📊 Amina's Invoice/Estimate Cost Coding Assistant")
st.markdown("Upload an invoice/estimate image to automatically assign cost codes and analyze expenses.")

# Initialize chat if not already done
if st.session_state.chat is None:
    st.session_state.chat = inspector.start_chat()

# File uploader
uploaded_file = st.file_uploader("Upload an invoice/estimate image", type=['png', 'jpg', 'jpeg'])

# Main interface
if uploaded_file:
    # Display image
    image = Image.open(uploaded_file)
    st.image(image, caption="Invoice/Estimate Image", use_container_width=True)
    
    # Analyze button
    if not st.session_state.image_analyzed:
        col1, col2 = st.columns([1, 2])
        with col1:
            if st.button("Analyze Invoice/Estimate", type="primary"):
                with st.spinner("Analyzing invoice/estimate and assigning cost codes..."):
                    # Store image for reference
                    st.session_state.current_image = image
                    
                    # Get initial analysis
                    report = inspector.analyze_image(image, st.session_state.chat)
                    
                    # Add the report to chat history
                    st.session_state.messages.append({"role": "assistant", "content": report})
                    st.session_state.image_analyzed = True
                    st.rerun()

# Display chat history
st.markdown("### 💬 Invoice/Estimate Analysis Chat")
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if st.session_state.image_analyzed:
    if prompt := st.chat_input("Ask questions about the invoice/estimate analysis..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                response = inspector.send_message(st.session_state.chat, prompt)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

# Footer with instructions
st.markdown("---")
st.markdown("""
### How to Use This Invoice/Estimate Analyzer
1. Upload a clear image of your construction invoice/estimate
2. Click "Analyze Invoice/Estimate" to get an automated cost code analysis
3. Review the generated cost codes and classifications
4. Ask questions about specific line items or classifications
5. Use the "Clear Chat History" button in the sidebar to start fresh

Example questions you can ask:
- Can you explain the reasoning for a specific cost code assignment?
- What items were flagged for review?
- How confident are you about the classifications?
- Can you break down a specific line item into sub-components?
- Are there any alternative cost codes that could apply to this item?
""")

# Download button for PDF report
if st.session_state.messages:
    # Create PDF buffer
    pdf_buffer = generate_pdf_report(
        st.session_state.messages, 
        st.session_state.current_image
    )
    
    # Download button for PDF
    st.download_button(
        "Download Analysis Report",
        pdf_buffer,
        file_name="invoice_analysis.pdf",
        mime="application/pdf"
    )