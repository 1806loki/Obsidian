-- Check total vs free space in ST_SENT_INDEXES_TS
SELECT
	df.tablespace_name,
	ROUND(df.total_mb, 2) AS "Total Allocated (MB)",
	ROUND(NVL(fs.free_mb, 0), 2) AS "Free Space (MB)",
	ROUND((df.total_mb - NVL(fs.free_mb, 0)), 2) AS "Used Space (MB)",
	ROUND(((df.total_mb - NVL(fs.free_mb, 0)) / df.total_mb) * 100, 2) AS "% Used"
FROM
	(
	SELECT
		tablespace_name,
		SUM(bytes)/ 1024 / 1024 AS total_mb
	FROM
		dba_data_files
	WHERE
		tablespace_name = 'ST_SENT_INDEXES_TS'
	GROUP BY
		tablespace_name) df
LEFT JOIN 
    (
	SELECT
		tablespace_name,
		SUM(bytes)/ 1024 / 1024 AS free_mb
	FROM
		dba_free_space
	WHERE
		tablespace_name = 'ST_SENT_INDEXES_TS'
	GROUP BY
		tablespace_name) fs
ON
	df.tablespace_name = fs.tablespace_name;

SELECT
	tablespace_name,
	file_name,
	bytes / 1024 / 1024 AS size_mb
FROM
	dba_data_files
WHERE
	tablespace_name = 'ST_SENT_INDEXES_TS';
-- Replace with actual file name if known
ALTER DATABASE DATAFILE '/rdsdbdata/db/ORCL_A/datafile/o1_mf_st_sent__mkh92kst_.dbf' RESIZE 500 M;

GRANT DBA TO ADMIN;

SELECT
	*
FROM
	SESSION_PRIVS
WHERE
	PRIVILEGE = 'ALTER DATABASE';

GRANT ALTER DATABASE TO ADMIN;

SELECT
	*
FROM
	dba_sys_privs
WHERE
	grantee = 'ADMIN';

ALTER TABLESPACE ST_SENT_INDEXES_TS 
ADD DATAFILE '/rdsdbdata/db/ORCL_A/datafile/o1_mf_st_sent_index__mkh92kst_.dbf' 
SIZE 1 G 
AUTOEXTEND ON 
NEXT 100 M 
MAXSIZE UNLIMITED;

ALTER TABLESPACE ST_SENT_INDEXES_TS 
ADD DATAFILE SIZE 200 M AUTOEXTEND ON;

 