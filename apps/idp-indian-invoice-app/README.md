# Invoice IDP Analytics (Databricks App)

Streamlit dashboard over the invoice extraction pipeline output in Unity Catalog
(`workspace.idp`): revenue, buyers, products, market basket, segments, CLV,
anomalies, data-quality checks and invoice-level view.

## Files
| File | Purpose |
|---|---|
| `app.py` | Streamlit app |
| `app.yaml` | Run command and env (`DATABRICKS_WAREHOUSE_ID` from the `sql-warehouse` resource) |
| `manifest.yaml` | Declares the SQL warehouse resource |
| `requirements.txt` | Python dependencies |

## Data (read-only)
`workspace.idp.finance_invoices`, `buyer_segments`, `buyer_clv`, `line_item_anomalies`

## Deploy
1. Create a Databricks App and attach a SQL warehouse resource with key `sql-warehouse`.
2. Grant the app's service principal `USE CATALOG` on `workspace`, and `USE SCHEMA` + `SELECT` on `workspace.idp`.
3. Point the app at this folder and deploy. Pushing to GitHub does not redeploy; redeploy from the app page.

## Notes
- No credentials are stored in this repo; auth comes from the Databricks App environment.
- Segments, CLV and anomaly tables are produced by the notebook, not by the app. Re-run the notebook to refresh them.