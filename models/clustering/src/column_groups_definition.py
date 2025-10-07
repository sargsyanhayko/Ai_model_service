import json

def get_columns_definition(input_path, adg):
    # Read the JSON file
    with open(input_path, 'r') as file:
        input_data = json.load(file)

    # Optionally read configs.json if needed
    with open('configs.json', 'r') as file:
        conf = json.load(file)

    for key in input_data:
        if key in conf:
            conf[key] = input_data[key]

    conf['ADG_code'] = adg
    # Save updated configs.json
    with open('configs.json', 'w') as file:
        json.dump(conf, file, indent=4)

    columns_definition = {
        "sector_columns": input_data.get("sector_columns"),
        "employee_columns": input_data.get("employee_columns"),
        "product_columns": adg,
        "additional_columns": input_data.get("additional_columns"),
        "invoice_columns": input_data.get("invoice_columns"),
        "location_columns": input_data.get("location_columns"),
        "count_columns": input_data.get("count_columns"),
        "revenue_columns": input_data.get("revenue_columns"),
        "time_flags": input_data.get("time_flags"),
    }

    return columns_definition, conf
