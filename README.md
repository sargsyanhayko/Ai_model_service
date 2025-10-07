# Required Parameters Per Model

## 1. Clustering

**Description:** Parameters required to run clustering analysis.

**Required Parameters:**

| Parameter | Description | Example | User input | Optinol |
|-----------|-------------|---------|------------|---------|
| `year` | Analysis year | `"2024"` | True | True |
| `ADG_code` | List of product/service codes | `["2402.0","2710.0"]` | True | True |
| `plots_folder` / `path_to_plots` | Folder for saving plots | `"plots"` | False | False |
| `sector` | Sector information or group | `"Finance"`, `"Retail"`, etc. | True | True |
| `sector_columns` | List of sector-related columns | `"level1", "TURNOVERTAXREPORT_ACTIVITY_CODE", ...` | True | False |
| `employee_columns` | List of employee-related columns | `"AVG_N_EMPLOYEES", "EMPLOYEE_CHANGE", ...` | True | False |
| `product_columns` | List of product columns | `"1905.0", "56.1", "2402.0", ...` | True | False |
| `additional_columns` | Additional metric columns | `"var1", "var1_std", "var2", ...` | True | False |
| `invoice_columns` | Invoice-related columns | `"supplier_COUNT_mean", "supplier_COUNT_std", ...` | True | False |
| `TINs` | Dictionary of entities | `"01282006"`, `"00029448"`, etc. | True | True |
| `target_column` | Target metric column | `"productivity", "profitability"` | True | True |
| `histogram_column` | Column used for histogram plots | `"productivity", "profitability"` | False | False |
| `location_columns` | Columns representing locations | `["REGION_CODE"]`, `["CITY_CODE"]` | False | False |
| `count_columns` | Count-related columns | `["OIN_COUNT","CRN_COUNT"]` | False | False |
| `revenue_columns` | Revenue-related columns | `["A41"]`, | False | False |
| `time_flags` | Time flag columns | `["TIME_FLAG"]` | False | False |

**Example `input.json`:**

```json
{
  "year": "2024",
  "ADG_code": ["2402.0","2710.0"],
  "plots_folder": "plots",
  "user": "ematevosyan",
  "columns_level1": {
    "sector_columns": true,
    "employee_columns": true,
    "product_columns": true,
    "additional_columns": true,
    "invoice_columns": true,
    "location_columns": true,
    "count_columns": true,
    "revenue_columns": true,
    "time_flags": true
  }
}
```
## 2. Time Series Analysis

**Description:** Parameters required for time-based analysis.

**Required Parameters:**

| Parameter | Description | Example | User input | Optinol |
|-----------|-------------|---------|------------|---------|
| `start_date` | Start of analysis period | `"2023-01-01"` | True | False |
| `end_date` | End of analysis period | `"2023-12-30"` | True | False |
| `time_col` | Column representing time | `"RECEIPT_TIME_HOUR"` | True | False |
| `product_id_col` | Column representing product groups | `"PRODUCT_GROUP"` | True | False |
| `target_col` | Target metric column | `"QANTITY_sum"` | True | False |
| `ADG_code` / `adg` | Product/service codes | `[2710]` | True | False |
| `TINs` | Dictionary of entities | `{"01282006": {}}` | True | False |
| `time_range` | Range for time-based analysis | `["2023-12-01", "2024-03-01"]` | True | False |
| `plot_temporal` | Generate temporal plots | `true` | False | False |
| `plot_entity` | Generate entity-level plots | `true` | False | False |
| `category_score` | Minimum category score | `0.5` | True | False |


**Example `input.json`:**

```json
{
  "start_date": "2023-01-01",
  "end_date": "2023-12-30",
  "time_col": "RECEIPT_TIME_HOUR",
  "product_id_col": "PRODUCT_GROUP",
  "target_col": "QANTITY_sum",
  "ADG_code": [2402.0, 2710.0],
  "TINs": {"01282006": {}},
  "time_range": ["2023-12-01", "2024-03-01"],
  "user": "ematevosyan",
  "plot_temporal": true,
  "plot_entity": true,
  "category_score": 0.5
}
```
## 3. Complex Audits & Frauds

**Description:** Parameters required to run audit/fraud analysis.

**Required Parameters:**

| Parameter | Description | Example |  User input | Optinol |
|-----------|-------------|---------|-------------|---------|
| `TIN` | List of entity identifiers | `["01282006", "00029448", "00024444"]` | True |  False |
| `TIME_FLAGS` | Year or period of analysis | `"2023"` | True |  False |

**Example `input.json`:**

```json
{
  "TIN": ["01282006", "00029448", "00024444"],
  "TIME_FLAGS": "2023",
  "user": "ematevosyan"
}
# models_service
# models_service
# models_service
