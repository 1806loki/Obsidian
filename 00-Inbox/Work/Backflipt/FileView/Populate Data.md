
Check what are the columns that are needed

Write a script to create the data of required columns

Deploy it to the cron job

INSERT INTO XFBTRANSFER_H (
    EVENTID,               -- 130300401
    COREID,                -- 'CORE123456'
    PROTOCOLFILENAME,      -- 'invoice_march_2025.csv'
    SENDERID,              -- 'partner-system'
    RECEIVERID,            -- 'internal-processor'
    EVENTTIMESTAMP,        -- 1748509200000
    CYCLEID,               -- 'RqTbGvL9NyxKUFwZ2PHDjmQoRsXeVb'
    STATE,                 -- 'RECEIVED'
    APPLICATION,           -- 'NYL-AdvancedRoute'
    GROUPNAME,             -- 'FinanceBU'
    BUSINESSTAG,           -- 'HIGH_PRIORITY'
    CLIENTNAME,            -- 'ClientX'
    PARTNERNAME,           -- 'PartnerABC'
    SOURCEACCOUNT,         -- 'Account123'
    USERID,                -- 'user.partnerabc'
    FILEID,                -- 'FILE123'
    FILEDESCRIPTION,       -- 'Monthly Invoice File'
    FILENAMEPATTERN,       -- 'INV_*_2025.csv'
    IDVALUE,               -- 'INV123456'
    DIVISION,              -- 'NorthDivision'
    LOCATION               -- 'axway-st-2'
) VALUES (
    130300401,             -- EVENTID value
    'CORE123456',          -- COREID value
    'invoice_march_2025.csv', -- PROTOCOLFILENAME value
    'partner-system',      -- SENDERID value
    'internal-processor',  -- RECEIVERID value
    1748509200000,         -- EVENTTIMESTAMP value (timestamp in milliseconds)
    'RqTbGvL9NyxKUFwZ2PHDjmQoRsXeVb', -- CYCLEID value (example of a unique ID or token)
    'RECEIVED',            -- STATE value
    'NYL-AdvancedRoute',   -- APPLICATION value
    'FinanceBU',           -- GROUPNAME value
    'HIGH_PRIORITY',       -- BUSINESSTAG value
    'ClientX',             -- CLIENTNAME value
    'PartnerABC',          -- PARTNERNAME value
    'Account123',          -- SOURCEACCOUNT value
    'user.partnerabc',     -- USERID value
    'FILE123',             -- FILEID value
    'Monthly Invoice File',-- FILEDESCRIPTION value
    'INV_*_2025.csv',      -- FILENAMEPATTERN value
    'INV123456',           -- IDVALUE value
    'NorthDivision',       -- DIVISION value
    'axway-st-2'           -- LOCATION value
);


SENDER_IDS = [
    "Apple",
    "Google",
    "Ford",
    "MaxNewYorkLife",
    "Samsung",
    "Toyota",
    "Amazon",
    "Caterpillar",
    "Adidas",
    "Walmart",
]

RECEIVER_IDS = [
    "Microsoft",
    "IBM",
    "Oracle",
    "Cisco",
    "Intel",
    "Dell",
    "HP",
    "VMware",
    "Salesforce",
    "Adobe",
    "Netflix",
]

APPLICATIONS = [
    "NYL-AdvancedRoute",
    "DataTransferPro",
    "FileSync-Enterprise",
    "SecureTransfer-v2",
    "BusinessFlow-App",
    "DataBridge-Suite",
]

PARTNERS = [
    "PartnerABC",
    "VendorXYZ",
    "ClientCorp",
    "ExternalSys-A",
    "ThirdPartyB",
    "IntegrationHub",
]

Source ACCOUNTS = [
    "Account123",
    "ProdAccount001",
    "TestAccount456",
    "DevAccount789",
    "UAAccount999",
    "MainAccount",
]

DIVISIONS = [
    "NorthDivision",
    "SouthDivision",
    "EastRegion",
    "WestRegion",
    "CentralOps",
    "GlobalDivision",
]


BUSINESS_TAGS = [
    "BT_FINANCE",
    "BT_HR",
    "BT_LOGISTICS",
    "BT_OPERATIONS",
    "BT_LEGAL",
    "BT_COMPLIANCE",
    "BT_IT",
    "BT_MARKETING",
    "BT_SALES",
    "BT_CUSTOMER_SERVICE",
]



Client_names = [
Client_A_1,
Client_A_50,
Client_A_100,
Client_B_1,
Client_B_25,
Client_B_100
]


