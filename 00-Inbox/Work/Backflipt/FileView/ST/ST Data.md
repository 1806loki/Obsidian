#TransferView/SQLQueryGenerationAgent 

![[Sentinel-Test Data.pdf]]


#### Sentinel Data : 

![[SENTinelData.xlsx]]


Using Sequence of States to find the number of transfers and number of sent and received files 
#### Phase-1 (Ruby)

- ptPush - RECEIVED -> TO_EXECUTE -> SENT 
- ptPull - TO_EXECUTE -> RECEIVED -> AVAILABLE
- stInbound - RECEIVED -> POST_PROC -> POST_PROC
- stOutbound - RECEIVED -> AVAILABLE -> SENT

#### Phase-2 
- stOutbound_ph2 -   AVAILABLE -> SENT
- ptPush_ph2 -   SENT -> no POST_PROC
- ptPull_ph2 -   RECEIVED -> AVAILABLE
- ptPull_site_ph_2 -   RECEIVED -> POST_PROC
- stInbound_ph2 - no TO_EXECUTE -> RECEIVED -> POST_PROC

We cannot distinguish between ptPushPH1 and ptPullPH2 just by their state sequences because they are identical. This appears to be a limitation of this approach., We can figure out the total number of transfers, but we cannot get the exact number of sent and received events using this approach.

ptPullPH2-C31VG-0212.docx  RECEIVING  
ptPullPH2-C31VG-0212.docx  RECEIVED  
ptPullPH2-C31VG-0212.docx  POST_PROC  
ptPullPH2-C31VG-0212.docx  TO_EXECUTE  
ptPullPH2-C31VG-0212.docx  SENDING  
ptPullPH2-C31VG-0212.docx  SENT  
ptPullPH2-C31VG-0212.docx  POST_PROC

  
ptPushPH1C1YK1-0212.docx  RECEIVING  
ptPushPH1C1YK1-0212.docx  RECEIVED  
ptPushPH1C1YK1-0212.docx  POST_PROC  
ptPushPH1C1YK1-0212.docx  TO_EXECUTE  
ptPushPH1C1YK1-0212.docx  SENDING  
ptPushPH1C1YK1-0212.docx  SENT  
ptPushPH1C1YK1-0212.docx  POST_PROC


Queries to find the number of sent and received files using UP1

```sql  
_/*List of sent and received files*/_  
SELECT Count(*)  
FROM   (SELECT DISTINCT coreid,  
                        x.protocolfilename  
        FROM   xfbtransfer_h x  
        WHERE  x.location = 'fileview-st'  
               AND state IN ( 'SENT', 'RECEIVED' )  
               AND x.userparameter1 = 'E')  
  
_/* Sender And Receiver */_  
SELECT xh.originalsenderid,  
       xh.receiverid,  
       xh.state,  
       xh.userparameter1,  
       xh.protocolfilename  
FROM   xfbtransfer_h xh  
WHERE  xh.location = 'fileview-st'  
       AND state IN ( 'SENDING', 'AVAILABLE' )  
       AND xh.originalsenderid <> xh.receiverid;
```




#### Populate Data
- Parameters passed to the function : Number of files, Time period 

List of columns with dynamic values
- PROTOCOLFILENAME
-  ORIGINALSENDERID
- SENDERID
- RECEIVERID
- FINALRECEIVERID
- USERID
- SITE
- RETURNMESSAGE
- COREID
- EVENTDATETIME

STATE and UP1 sequence and values are already predefined set

**Prompt:**
I have a CSV file (sentinelData2 - Copy.xlsx - All Phases Data.csv) containing log records for various file transfers. Please write a Python script that analyzes this data and then inserts synthetic records into an Oracle database.

1. Pattern Recognition:

Load the CSV and group the data by COREID.

Identify the unique 'Transfer Types' (distinct sequences of events). For each type, extract the specific sequence of values for: ORIGINALSENDERID, SENDERID, RECEIVERID, FINALRECEIVERID, USERID, SITE, STATE, and UP1. These values must remain consistent for that specific type of transfer.

2. Database Insertion Function: Create a function insert_transfer_data(num_files, time_period_days, db_connection) that accepts the number of files to simulate, the time range, and an active Oracle database connection object. The function should perform the following logic:
3. Use the FileNa

Loop num_files times:

Select Template: Randomly pick one of the identified Transfer Types.

The Site Column value depends on USERID

Generate Identifiers: Create a unique COREID (UUID) and a PROTOCOLFILENAME. The filename should be prefixed (e.g., Type1_fileA) so the specific transfer type is identifiable.

Use The FILENAMEPATTERN As is for each type of file transfer

Set Time: Generate a random start time within the next time_period_days.

Process Sequence: Iterate through the ordered steps of the selected Transfer Type:

Use the fixed values from the template for the required fields.

Set EVENTDATETIME: Use the start time for the first step, and increment by a few seconds for each subsequent step to ensure chronological order.

Execute Insert: Construct and execute a parameterized SQL INSERT statement using the db_connection cursor to write the row directly to the database table.

