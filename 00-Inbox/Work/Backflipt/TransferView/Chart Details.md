Chart Details :

Bar Chart : 


Request :

{
    "question": "Are there any failed transfers today?",
    "query": "SELECT COUNT(*) AS \"FAILED TRANSFERS\", SENDERID AS \"SENDER ACCOUNT\" FROM XFBTRANSFER_H xh WHERE State = '\''FAILED'\'' AND EventTimestamp >= ((TRUNC(SYSDATE)) - TO_DATE('\''1970-01-01'\'', '\''YYYY-MM-DD'\'')) * 86400000 GROUP BY SENDERID ORDER BY COUNT(*) DESC",
    "chartData": {
        "FAILED TRANSFERS": 31,
        "SENDER ACCOUNT": "localhost:22"
}
 

Response :

{
    "chart_type": "Bar chart",
    "data_points": {
        "x-axis": "SENDER ACCOUNT",
        "y-axis": "FAILED TRANSFERS"
    },
    "reason": "A bar chart is the most appropriate visualization for this data as it effectively compares the number of failed transfers across different sender accounts. Each bar will represent a sender account, with the height indicating the number of failed transfers, making it easy to identify which accounts have experienced the most failures today. This visualization allows for quick identification of problematic sender accounts that may require attention."
}


Pie Chart
 
 Request :
 
{
    "question": "Give me all the success and failed file transfers seperately",
    "query": "SELECT \n    'Success' AS \"TRANSFER_STATUS\", \n    COUNT(*) AS \"COUNT\" \nFROM \n    XFBTRANSFER_H \nWHERE \n    State IN ('AVAILABLE', 'SENT', 'POST_PROC/ROUTED', 'POST_PROC/ARCHIVED') \n    AND EventTimestamp >= ((TRUNC(SYSDATE) - 1 - DATE '1970-01-01') * 86400000)\n    AND EventTimestamp < ((TRUNC(SYSDATE) - DATE '1970-01-01') * 86400000)\nUNION ALL\nSELECT \n    'Failed' AS \"TRANSFER_STATUS\", \n    COUNT(*) AS \"COUNT\" \nFROM \n    XFBTRANSFER_H \nWHERE \n    State IN ('FAILED', 'POST_PROC') \n    AND EventTimestamp >= ((TRUNC(SYSDATE) - 1 - DATE '1970-01-01') * 86400000)\n    AND EventTimestamp < ((TRUNC(SYSDATE) - DATE '1970-01-01') * 86400000)",
    "chart_data": {"TRANSFER_STATUS": "Success", "COUNT": 366},
}
 
 
 
Response : 
 
{
    "chart_type": "Pie chart",
    "data_points" : {'columns': ['TRANSFER_STATUS', 'COUNT']},      
"reason": "A pie chart is the most appropriate visualization for this data as it clearly shows the proportion of successful versus failed file transfers. The data contains categorical status information (Success/Failed) and corresponding count values, making it ideal for displaying part-to-whole relationships. This visualization will provide an immediate visual understanding of the success rate of file transfers."
}
    

Line Chart 
 
Request : 
 
 
{
    "question": "Show me count of file transfers with daily data on every month for last 6 months",
    "query": "SELECT EXTRACT(MONTH FROM (TO_DATE('1970-01-01', 'YYYY-MM-DD') + (EVENTTIMESTAMP / 86400000))) AS \"MONTH\", EXTRACT(DAY FROM (TO_DATE('1970-01-01', 'YYYY-MM-DD') + (EVENTTIMESTAMP / 86400000))) AS \"DAY\", COUNT(EVENTID) AS \"NUMBER OF TRANSFERS\" FROM XFBTRANSFER_H WHERE EVENTTIMESTAMP >= ((TRUNC(ADD_MONTHS(SYSDATE, -6), 'MM') - TO_DATE('1970-01-01', 'YYYY-MM-DD')) * 86400000) AND EVENTTIMESTAMP < ((LAST_DAY(ADD_MONTHS(SYSDATE, -1)) + INTERVAL '1' DAY - TO_DATE('1970-01-01', 'YYYY-MM-DD')) * 86400000) GROUP BY EXTRACT(MONTH FROM (TO_DATE('1970-01-01', 'YYYY-MM-DD') + (EVENTTIMESTAMP / 86400000))), EXTRACT(DAY FROM (TO_DATE('1970-01-01', 'YYYY-MM-DD') + (EVENTTIMESTAMP / 86400000))) ORDER BY EXTRACT(MONTH FROM (TO_DATE('1970-01-01', 'YYYY-MM-DD') + (EVENTTIMESTAMP / 86400000))) ASC, EXTRACT(DAY FROM (TO_DATE('1970-01-01', 'YYYY-MM-DD') + (EVENTTIMESTAMP / 86400000))) ASC",
    "chart_data": {"MONTH": 1, "DAY": 1, "NUMBER OF TRANSFERS": 48},
}
 
 
Resposne :
 
{
    "chart_type": "Line chart",
    "data_points": {
        "x-axis": "DAY",
        "y-axis": "NUMBER OF TRANSFERS",
        "line-series": "MONTH"
    },
    "reason": "A line chart is the most appropriate visualization for this data as it effectively shows the daily trend of file transfers across multiple months. The data contains daily counts (DAY) for each MONTH with corresponding NUMBER OF TRANSFERS values, making it perfect for a time-series visualization. This chart will allow users to easily identify patterns, peaks, and valleys in transfer activity throughout each month, as well as compare transfer volumes between different months over the 6-month period."
}