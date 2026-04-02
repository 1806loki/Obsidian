[SecureTransport 5.5 Administrator Guide](https://docs.axway.com/bundle/SecureTransport_55_AdministratorGuide_allOS_en_HTML5/page/Content/AdministratorsGuide/STAdminToolStartPage.htm)


**ZeroByteWildcardPullAllowed**
A zero-byte entry may indicate either a failed transfer (shown in red) or a successful SIT pull with no files returned. [(SecureTransport Glossary)](https://docs.axway.com/bundle/SecureTransport_55_AdministratorGuide_allOS_en_HTML5/page/Content/ST_glossary.htm#Logging_)

#### File Tracking Fields Definitions (based on industry standards for file transfer tracking systems and MFT (Managed File Transfer) platforms)

| Field                     | Meaning                                                                              |
| ------------------------- | ------------------------------------------------------------------------------------ |
| **Status**                | The transfer status (e.g., Success, Failed, Pending, In Progress)                    |
| **Account**               | The user account associated with the file transfer                                   |
| **Login**                 | The login name/username used to authenticate the transfer                            |
| **UserClass**             | Classification/category of the user (e.g., Standard, Partner, Service Account)       |
| **UserType**              | Type of user account (e.g., Human, System/Service)                                   |
| **Application**           | The application or system initiating the transfer                                    |
| **Direction**             | Transfer direction (Upload/Send or Download/Receive)                                 |
| **Action By**             | The user or process that initiated/performed the action                              |
| **Size**                  | File size in bytes or human-readable format                                          |
| **File**                  | The filename being transferred                                                       |
| **Mode**                  | Transfer mode (ASCII, Binary, or protocol-specific mode)                             |
| **Transfer Site**         | The transfer site/endpoint involved in the transfer                                  |
| **Transfer Content Type** | MIME type or content classification of the file                                      |
| **Remote Folder**         | The remote directory/folder path where the file resides                              |
| **Local Filename**        | The local filename as stored locally                                                 |
| **Local Folder**          | The local directory/folder path                                                      |
| **ICAP Details**          | Internet Content Adaptation Protocol details (if antivirus/content scanning enabled) |
| **Local File**            | The local file path and name                                                         |
| **Protocol**              | Protocol used (SFTP, FTPS, AS2, PeSIT, HTTP/HTTPS, etc.)                             |
| **Secure**                | Security status (Encrypted, Signed, or security method used)                         |
| **Start Time**            | Timestamp when the transfer started                                                  |
| **End Time**              | Timestamp when the transfer completed                                                |
| **Duration**              | Total transfer duration in seconds or time format                                    |
| **Remote Host**           | IP address or hostname of the remote system                                          |
| **Transfer ID**           | Unique identifier for the file transfer transaction                                  |
| **Session ID**            | Unique identifier for the user/connection session                                    |
| **Session Start Time**    | When the session was established                                                     |
| **Operation Index**       | Sequential index for operations within a session                                     |
| **Pesit Message**         | Message or status related to PeSIT protocol transfers                                |
| **CoreId**                | Internal system identifier for correlation/tracking                                  |
| **Resubmitted**           | Indicates if the file was resubmitted after a previous failure                       |
| **Additional Info**       | Extra metadata or custom information about the transfer                              |
| **X-Forwarded-For**       | Original client IP address (used when behind proxy)                                  |
| **SecurityParameters**    | Applied security settings (encryption algorithms, signing methods, etc.)             |
| **Server Name**           | Name/identifier of the SecureTransport server processing the transfer                |