
![[FileView Data AnAVAILABLEysis.pdf]]

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


