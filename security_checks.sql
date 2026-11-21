-- DATABASE SECURITY TESTING
-- 1. DATABASE PRIVILEGE CHECK
-- Purpose: Checks the current user's database permissions.

SELECT
    current_database() AS database_name,
    current_user AS connected_role,
    pg_get_userbyid(d.datdba) AS database_owner,
    has_database_privilege(
        current_user, current_database(), 'CONNECT'
    ) AS can_connect,
    has_database_privilege(
        current_user, current_database(), 'CREATE'
    ) AS can_create,
    has_database_privilege(
        current_user, current_database(), 'TEMP'
    ) AS can_create_temp_tables
FROM pg_database d
WHERE d.datname = current_database();

-- 2. SCHEMA PRIVILEGE CHECK
-- Purpose: Checks permissions on the public schema.


SELECT
    current_user AS connected_role,
    has_schema_privilege(
        current_user, 'public', 'USAGE'
    ) AS can_use_schema,
    has_schema_privilege(
        current_user, 'public', 'CREATE'
    ) AS can_create_objects;


-- 3. TABLE OWNERSHIP CHECK
-- Purpose: Identifies the owner of each project table.


SELECT
    tablename AS table_name,
    tableowner AS table_owner
FROM pg_tables
WHERE schemaname = 'public'
ORDER BY tablename;

-- 4. PUBLIC SCHEMA ACCESS CHECK
-- Purpose: Checks whether all database roles
-- have access to the public schema.

SELECT
    nspname AS schema_name,
    COALESCE(grantee.rolname, 'PUBLIC') AS grantee,
    privilege_type
FROM pg_namespace n
CROSS JOIN LATERAL aclexplode(
    COALESCE(n.nspacl, acldefault('n', n.nspowner))
) AS acl
LEFT JOIN pg_roles grantee
    ON grantee.oid = acl.grantee
WHERE n.nspname = 'public'
ORDER BY grantee, privilege_type;

-- 5. ROW LEVEL SECURITY CHECK
-- Purpose: Checks whether Row-Level Security (RLS)
-- is enabled or forced on project tables.

SELECT
    c.relname AS table_name,
    c.relrowsecurity AS rls_enabled,
    c.relforcerowsecurity AS rls_forced
FROM pg_class c
JOIN pg_namespace n
    ON n.oid = c.relnamespace
WHERE n.nspname = 'public'
AND c.relkind = 'r'
ORDER BY c.relname;

-- 6. PASSWORD STORAGE CONFIGURATION CHECK
-- Purpose: Checks PostgreSQL password encryption setting.

SHOW password_encryption;

-- 7. DATABASE CONNECTION SECURITY CHECK
-- Purpose: Checks SSL usage for active connections.

SELECT
    a.usename AS username,
    a.datname AS database_name,
    s.ssl,
    s.version AS ssl_version
FROM pg_stat_activity a
JOIN pg_stat_ssl s
    ON a.pid = s.pid
WHERE a.datname = current_database();

-- 8. DATABASE LOGGING CONFIGURATION CHECK
-- Purpose: Checks PostgreSQL connection and
-- authentication logging settings.

SHOW log_connections;

SHOW log_disconnections;

SHOW log_line_prefix;

-- 9. TABLE ACCESS PERMISSIONS CHECK
-- Purpose: Checks which database roles have been
-- granted permissions on project tables.

SELECT
    c.relname AS table_name,
    COALESCE(r.rolname, 'PUBLIC') AS grantee,
    a.privilege_type,
    a.is_grantable
FROM pg_class c
JOIN pg_namespace n
    ON n.oid = c.relnamespace
CROSS JOIN LATERAL aclexplode(
    COALESCE(c.relacl, acldefault('r', c.relowner))
) AS a
LEFT JOIN pg_roles r
    ON r.oid = a.grantee
WHERE n.nspname = 'public'
  AND c.relkind = 'r'
  AND (
      a.grantee = 0
      OR a.grantee <> (
          SELECT oid
          FROM pg_roles
          WHERE rolname = 'postgres'
      )
  )
ORDER BY c.relname, grantee, a.privilege_type;

-- 10. DATABASE ROLE SECURITY CHECK
-- Purpose: Checks login roles and their privileges.

SELECT
    rolname AS role_name,
    rolsuper AS is_superuser,
    rolcreatedb AS can_create_database,
    rolcreaterole AS can_create_roles,
    rolcanlogin AS can_login
FROM pg_roles
WHERE rolcanlogin = TRUE
ORDER BY rolname;

