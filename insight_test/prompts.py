# Analysis prompts for different document types

PDF_PROMPT = """Analyze this financial document and provide:
1. Key financial metrics:
   - Extract and verify all numerical values
   - Convert text-based numbers to numerical format
   - Identify and standardize units (thousands, millions, etc.)

2. Financial health assessment:
   - Analyze current financial position
   - Compare against industry standards
   - Identify trends and patterns
   - Evaluate operational efficiency

3. Risk assessment:
   - Identify potential red flags
   - Analyze debt structure and obligations
   - Evaluate market and industry risks
   - Consider regulatory compliance issues

4. Strategic recommendations:
   - Provide actionable insights
   - Suggest areas for improvement
   - Recommend risk mitigation strategies
   - Outline potential growth opportunities

5. Creditworthiness evaluation:
   - Calculate key financial ratios
   - Compare to industry benchmarks
   - Consider qualitative factors
   - Provide a score from 0-100 with detailed justification

Format as JSON with structure:
{
    "metrics": {
        "revenue": number,
        "net_income": number,
        "total_assets": number,
        "total_liabilities": number,
        "cash_flow": number
    },
    "health_assessment": "string",
    "risk_factors": ["string"],
    "recommendations": ["string"],
    "credit_score": number,
    "ratios": {
        "debt_to_equity": number,
        "current_ratio": number,
        "quick_ratio": number
    },
    "analysis_confidence": number
}"""

ENGINEERING_CSV_PROMPT = """Analyze this robot joint data and provide:
1. Individual J6 Analysis:
   - Calculate average current/speed ratio for each joint
   - Identify min/max current/speed ratios
   - Compare ratios across operating speeds

2. Friction Analysis:
   - Calculate absolute current/speed ratios
   - Identify any significant variations
   - Note any concerning patterns

3. Anomaly Detection:
   - Flag any unusual current/speed relationships
   - Identify joints with abnormal friction
   - Quantify deviations from expected behavior

Format the response as a complete, valid JSON object with this exact structure:
{
    "data_overview": {
        "analyzed_arms": ["arm900", "arm901", "arm902", "arm903", "arm906", "arm907", "arm909"],
        "record_count": number
    },
    "joint_data": {
        "arm900": [{"speed": number, "current": number}],
        "arm901": [{"speed": number, "current": number}],
        "arm902": [{"speed": number, "current": number}],
        "arm903": [{"speed": number, "current": number}],
        "arm906": [{"speed": number, "current": number}],
        "arm907": [{"speed": number, "current": number}],
        "arm909": [{"speed": number, "current": number}]
    },
    "technical_insights": {
        "behavior": [
            "Average current/speed ratios per joint",
            "Notable friction variations",
            "Specific anomalies detected"
        ],
        "optimization": [
            "Joints requiring attention",
            "Detailed friction analysis"
        ]
    }
}

Important:
1. Include ALL data points in joint_data for plotting
2. Do not use ellipsis (...) or comments in the JSON
3. Ensure the JSON is complete and valid
4. Keep the exact structure shown above"""




FINANCIAL_CSV_PROMPT = """Analyze this CSV data and provide:
1. Data Overview:
   - Number of records
   - Key columns identified
   - Data quality assessment

2. Statistical Analysis:
   - Summary statistics
   - Key trends
   - Notable patterns

3. Financial Metrics:
   - Calculate relevant financial ratios
   - Identify key performance indicators
   - Track changes over time

4. Recommendations:
   - Data-driven insights
   - Areas for improvement
   - Action items

Format as JSON with structure:
{
    "data_overview": {
        "record_count": number,
        "columns": [string],
        "quality_score": number
    },
    "statistics": {
        "summary": object,
        "trends": [string],
        "patterns": [string]
    },
    "metrics": {
        "ratios": object,
        "kpis": object
    },
    "recommendations": [string]
}"""

TXT_PROMPT = """Analyze this text document and provide:
1. Content Analysis:
   - Main topics
   - Key points
   - Important dates/numbers

2. Financial Information:
   - Extract monetary values
   - Identify financial terms
   - Find relevant dates

3. Risk Assessment:
   - Potential issues
   - Areas of concern
   - Compliance matters

4. Action Items:
   - Required steps
   - Follow-up tasks
   - Recommendations

Format as JSON with structure:
{
    "content": {
        "topics": [string],
        "key_points": [string],
        "important_data": object
    },
    "financial_info": {
        "monetary_values": object,
        "terms": [string],
        "dates": [string]
    },
    "risks": {
        "issues": [string],
        "concerns": [string],
        "compliance": [string]
    },
    "actions": [string]
}"""

DOCX_PROMPT = """Analyze this document and provide:
1. Document Structure:
   - Sections identified
   - Key headings
   - Important paragraphs

2. Content Analysis:
   - Main points
   - Critical information
   - Supporting details

3. Financial Data:
   - Monetary values
   - Financial terms
   - Calculations

4. Recommendations:
   - Key takeaways
   - Action items
   - Follow-up steps

Format as JSON with structure:
{
    "structure": {
        "sections": [string],
        "headings": [string],
        "key_paragraphs": [string]
    },
    "content": {
        "main_points": [string],
        "critical_info": object,
        "details": [string]
    },
    "financial": {
        "values": object,
        "terms": [string],
        "calculations": object
    },
    "recommendations": [string]
}"""

# Dictionary mapping file extensions to their prompts
ANALYSIS_PROMPTS = {
    '.pdf': PDF_PROMPT,
    '.csv': ENGINEERING_CSV_PROMPT,  # Default to engineering analysis for CSV
    '.txt': TXT_PROMPT,
    '.docx': DOCX_PROMPT
}

# Function to get the appropriate prompt
def get_prompt(file_ext, data_type='engineering'):
    """
    Get the appropriate analysis prompt based on file extension and data type.
    
    Args:
        file_ext (str): File extension (e.g., '.csv', '.pdf')
        data_type (str): Type of data ('engineering' or 'financial')
    
    Returns:
        str: The analysis prompt to use
    """
    if file_ext == '.csv':
        return ENGINEERING_CSV_PROMPT if data_type == 'engineering' else FINANCIAL_CSV_PROMPT
    return ANALYSIS_PROMPTS.get(file_ext)
