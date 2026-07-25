# Permission matrix

`Allow` means the action is permitted only inside the authenticated airport.
Explicit Watch Manager permissions are stored per membership/user.

| Capability | Super Admin | Unit Admin | Roster Editor | Watch Manager | Staff User | Read-only Auditor |
| --- | --- | --- | --- | --- | --- | --- |
| Airport plan/status/limits | Allow, aggregates only | View own usage | No | No | No | No |
| Personnel and operational roster | No | Allow | Allow | Explicit | Own view | Target only; not provisionable |
| Edit roster assignments | No | Allow | Allow | `edit_roster` | No | No |
| Create own shift request | No | Own unit only | Own only | Own only | Own only | No |
| Decide/apply shift request | No | Allow | No | No | No | No |
| Manage annotation definitions | No | Allow | No | No | No | No |
| Apply permitted annotations | No | Allow | Allow | `apply_annotations` | No | No |
| Bulk annotations / TOIL | No | Allow | Allow if permitted | Explicit | No | No |
| Manage qualifications/rules | No | Allow | No | No | No | No |
| Publish/rollback roster | No | Allow | No | No | No | No |
| Acknowledge publication | No | Own membership | Own membership | Own membership | Own membership | No |
| Invite/deactivate logins | No | Allow within limit | No | No | No | No |
| Platform feature flags/suspension | Allow, allowlist only | No | No | No | No | No |
| Personnel exports/impersonation | No | Unit-scoped exports only | Role-scoped | Role-scoped | Own only | No |

Denied routes must return 403 or conceal non-owned resources with 404. A role
label never overrides unit ownership. Super Admin has no universal operational
access.

