
### **Release Deployment Steps**

#### **1. Create a Jira Ticket**

**Ticket Title:**  
`TIQ Release - v3.4.0 - Dev Apps - 19th August 2025`

**Release Notes:**  
[TIQ v3.4.0 Release Report](https://backflipt1.atlassian.net/projects/TIQ/versions/14322/tab/release-report-all-issues)

**Environment:**  
`DEV-APPS`

---

#### **2. Build and Push Docker Images**

**TIQ Image**

- **Tag:** `dockerbackflipt/tiq-fileview:v3.4.0`
    

**Commands:**

```bash
docker build --platform=linux/amd64 -t dockerbackflipt/tiq-fileview:v3.4.0 .
docker push dockerbackflipt/tiq-fileview:v3.4.0
docker run -p 8080:8080 dockerbackflipt/tiq-fileview:v3.4.0
```

---

**TAACOs Image**

- **Tag:** `dockerbackflipt/taacos:v1.0.0`
    

**Commands:**

```bash
docker build --platform=linux/amd64 -t dockerbackflipt/taacos:v1.0.0 .
docker push dockerbackflipt/taacos:v1.0.0
docker run -p 8080:8080 dockerbackflipt/taacos:v3.4.0
```

---

#### **3. Create a Teams Post**

Once both the Jira ticket and Docker images are ready, post the following in Teams:

**Message Template:**

> **Release Request:**  
> **Ticket:** TIQ Release - v3.4.0 - Dev Apps - 19th August 2025  
> **Docker Image:** dockerbackflipt/tiq-fileview:v3.4.0  
> **Environment:** DEV-APPS  
> **Release Notes:** [Click here](https://backflipt1.atlassian.net/projects/TIQ/versions/14322/tab/release-report-all-issues)
> 
> Requesting approval to proceed with the **TIQ v3.4.0** release.
