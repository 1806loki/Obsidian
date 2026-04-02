Alerts Revamp

Prompt Changes : 
1. Need to add few shots json as template variables and remove the few shots included in the prompt.
2. Make changes in the strcutured response, define the fields in the structured response.
3. Implement with_structured_output 
4. Change Notification Prompt Structure based on the ticket - 982
5. Make changes in the prompt to handle the filetransfertime, alertTimetype, fileSearchType and fileTransferCondition

Present Notification Prompt Structure

<role>
<objective>
<instructions>
<response_format>
<mandatory_fields>
<field_notes>
<examples>
<example>
<example>
</examples>
<question_input>
<output_instruction>

New Notifications Prompt Strcuture

<Context>
  <Role>
  <Objective>
  <Input>
  <Tone>
  <OutputFormatInstructions>
  <ReasoningGuidelines>
  <ResponseInstructions>
</Context>

Notification Config

Present Notification Config

{
        "partner": "ABC",
        "file_count": 1,
        "business_unit": "",
        "file_count_comparator": ">=",
        "file_name_pattern": "%.pdf",
        "schedule_config": {
            "schedule_type": "Custom", //daily , weekly , monthly
            "time": "14:26",
            "day_of_week": "",
            "date_of_month": "",
            "run_type": "AT"
        },
        "event_type": "NotReceived",
        "email_list_to_send": [
            "ops@company.com",
            "sample@dmoain.com"
        ],
        "file_id": "",
        "business_tags": "",
        "data_sensitivity": "",
        "client_name": ""
}

New Notification Config

{
    "emails": ["sample@domain.com, sample2@domain.com"],
    "alert_config": {
        "alert_condition_check_frequency": "Daily",
        "alert_time": "17:10",
        "alert_time_type": "at",
        "timezone": "America/New_York",
        "cron_time": "0 0 1 1 *",
        "date_of_month": "06/05/2025",
        "day_of_week": "Monday",
    },
    "file_config": {
        "file_transfer_condition": "NotReceived",
        "file_count": 4,
        "file_name": "sample.txt",
        "file_search_type": "EXACT",
        "file_transfer_start_time": "10:00",
        "file_transfer_end_time": "12:00",
        "file_transfer_time_type": "before",
        "file_count_comparator": "==",
        "accounts": ["sample1", "sample2"],
        "accounts_receivers": ["sample3", "sample4"],
        "business_unit": "unit",
        "client_name": "client",
        "business_tag": "tag",
        "partners": ["Google", "Zebronics"],
        "partner_account": ["PartnerAccount-1"]
    }
}


Notifications flow in Platform

Flows

- AI-V2: Notifications Scheduler

- AI-V2: Generate Notifications

Business logic

- Ask/Update question(conversation)

- Notification-Flow

- Notifcations: Fetch Data

Notify every Tuesday at 12:25 PM i when partner Partner1 and Partner2 send at least 15 files with the name pattern invoice_.pdf  to accounts upoijon123, thiopj456 and business unit Finance, partner accounts poinso234, eden989 and Operations in the past 30 days.  Include files that failed to transfer. The alert should be sent to notify@domain.com, alerts@domain.com,and support@domain.com. Add business tags Urgent , and set the client names as ClientABC. For sent files, monitor account1, account2 and client789 for failures in receiving files.

few shot :
{
    "question": "Alert me on the last day of every month at 6 PM if client ReportingSystem sends more than 100 files between 12 AM and 6 AM with business tag EOD.",
    "notificationData": {
      "emails": [],
      "alert_config": {
        "alert_time": "18:00",
        "alert_time_type": "at",
        "day_of_week": "",
        "date_of_month": "last",
        "alert_condition_check_frequency": "Monthly",
        "time_zone": "America/New_York",
        "cron_time": "0 18 L * *"
      },
      "file_config": {
        "file_transfer_condition": "Sent",
        "file_count": "100",
        "file_name": "",
        "file_search_type": "",
        "file_transfer_start_time": "00:00",
        "file_transfer_end_time": "06:00",
        "file_transfer_time_type": "day",
        "file_transfer_time_period": "",
        "file_count_comparator": ">",
        "accounts": [],
        "accounts_receivers": [],
        "business_unit": "",
        "client_name": "ReportingSystem",
        "business_tag": "EOD",
        "partners": [],
        "partner_accounts": []
      }
    },
    "uniqRefId": "e5f6a7b8-c9d0-1234-ef56-567890123456"
  },


Cases need to handled :

1. alert me last of every month
2. Alert me when there is a file transfer on sunday
3. Alert me only today


Alert me daily at 9 AM if partner ABC has not sent any files between 8 AM and 6 PM.

Send weekly alerts every Friday at 5 PM to check if accounts Google and Microsoft have received at least 5 files this week so far.

Check monthly on the 15th at 2:30 PM if client ACME has sent more than 10 files in the last 3 months to account DataCenter.

Monitor every 5 minutes if business unit Sales has received files matching pattern .csv from partner DataProvider, alert at 11:45 AM.

Alert me daily at 8 AM if no files were transferred from account MainServer yesterday.

Send alerts weekly on Mondays at 10 AM if files tagged as 'Urgent' were not received between 9 PM and 11 PM in Pacific timezone.

Check daily at 6:30 PM if partners PartnerA and PartnerB have sent the file 'daily_report.xlsx' today, and alert if less than 2 files received.

Alert monthly on the last day at 3 PM if business unit Finance has not sent any files this quarter so far to partner accounts PA1 and PA2.

Send alerts every Tuesday at 1 PM if account ServerA has received exactly 1 file named 'backup_*.log' between 2 AM and 4 AM in the past week.

Monitor daily at 7:45 AM if client BigCorp from business unit Operations has failed to transfer any files to accounts Receiver1 and Receiver2 after 6 PM yesterday.

Alert me if account TestAccount has not received any files.

Check weekly if partner XYZ has sent files.

Send alerts if no files were transferred from MainServer.

Monitor if client ABC receives files on Mondays.

Alert at 5 PM if files are not sent.

Alert me daily if account Sender1 has sent files to account Receiver1 and partner Receiver2.

Check weekly if partner PartnerA from business unit Marketing has received files from account ClientServer tagged as 'Priority'.

Monitor monthly if client BigClient has sent files to partner accounts PA1, PA2, and PA3 in the last quarter.

Alert daily if business unit Operations has received files from partners Google and Microsoft to accounts DB1 and DB2.

Send alerts if account Server1 has transferred files to client TestClient via partner accounts ConnectorA and ConnectorB.

Alert me if more than 100 files were sent from account HighVolume in the last 2 weeks.

Monitor daily if less than 3 files named 'report_*.txt' were received by account Analytics between 1 AM and 3 AM.

Alert me if all files are Failed with the business tag Urgent in the last 7 days.

Notify me if more than 5 files are Sent for client ACME this week.

Notify me if a file is Received for partner account PartnerAccount1.

Notify me if at least one file is Sent for partner Amazon and business tag Marketing.

Notify me if more than 10 files are transferred for client TechCorp this month.

Notify me if no files are received from partner Walmart in the last 30 days. Count should be equal to 0.

Notify me if the total number of Sent files exceeds 50 in the last 6 months for business tag Finance.

Notify me if the file count falls below 5 for partner account PartnerAccount2.

Notify me if the file count equals zero for business tag HR and partner Google.

Notify me if any Failed occurs for partner Microsoft, partner account PartnerAccount1, business tag Legal, and client XYZ Corp.

Notify me if no Received files are found for business tag Operations, partner Amazon, and client ACME Corp. Count should be equal to 0.

Notify me if the Received file count for partner Google and business tag Finance is greater than 100.

Alert me daily if no files were transferred for client TechCorp yesterday.

Alert me daily if no files were transferred for client Amazon with business tag Marketing.

Alert weekly if files are sent from client XYZ Corp to partners Amazon and Google.

Notify me if no successful file transfers occurred from account XYZ in the last 30 days.



Alert Time
- Alert me at 7:30 AM every day if any file is received."
- Notify me before 5 PM if no files are sent today."

Alert Frequency (Daily, Weekly, Monthly, Custom)
- "Alert me every Monday at 9 AM if any file is received from partner ACME."
- "Alert me every 5 minutes if no files are received from client DataCorp."

 Day of Week / Date of Month
- "Notify me every Friday at 3 PM if no files are received from partner XYZ."
- "Alert me on the 15th of every month at 10:00 if less than 5 files are sent."

 File Transfer Condition
- "Notify me if any file is received from partner ACME."
- "Alert me if no files are sent from account Sender2."

 File Count & Comparator
- "Alert me if more than 10 files are sent by account Sender1."
- "Notify me if exactly 5 files are sent from client TechCorp."

 File Name & Search Type
- "Notify me if the file report.csv is not sent from account DataSource."
- "Alert me if any file matching *.xml is received from partner XMLPartner."

  File Transfer Start/End Time
- "Alert me if files are sent from account Sender1 between 9 AM and 5 PM."
- "Notify me if files are received from partner ACME before 8 AM."

 File Transfer Time Period & Type
- "Notify me if no files are received from partner XYZ in the last 7 days."
- "Alert me if exactly one file is sent from partner account PartnerAcc1 in the last quarter."

  Accounts & Accounts Receivers
- "Alert me if more than 5 files are sent by account Sender1 to account Receiver1."

 Partners & Partner Accounts
- "Notify me if any file is received from partner ACME."
- "Notify me if the file report.csv is not sent from account DataSource to partner account PA123 before 2 PM."

 Dynamic Time Periods
- "Alert me if partner TechVendor has not sent any files this week so far."
- "Notify me if client DataMart receives more than 20 files from the start of this month till now with business tag MONTHLY."




Changes : 

1. File transfer defintion should be moved to different section



Notify lokesht@backflipt.com at monthly 30th 4:59 PM if there are than 1000 file transfers in the last 7 days




