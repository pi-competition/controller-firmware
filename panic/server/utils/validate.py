def validate(schema, content):
    errors = []
    for item in schema:
        if item["required"] == True and item["name"] not in content:
            errors.append(f"Missing required field {item['name']}")
        if item["name"] in content:
            print(type(content[item["name"]]).__name__)
            if item["type"] == "str":
                if type(content[item["name"]]).__name__ != "str":
                    errors.append(f"Field {item['name']} must be a string")
                    continue 
                content[item["name"]] = content[item["name"]].strip()
                if "min" in item and len(content[item["name"]]) < item["min"]:
                    errors.append(f"Field {item['name']} must be at least {item['min']} characters")
                if "max" in item and len(content[item["name"]]) > item["max"]:
                    errors.append(f"Field {item['name']} must be at most {item['max']} characters")
            if item["type"] == "int":
                if type(content[item["name"]]).__name__ != "int":
                    errors.append(f"Field {item['name']} must be an integer")
                if "min" in item and content[item["name"]] < item["min"]:
                    errors.append(f"Field {item['name']} must be at least {item['min']}")
                if "max" in item and content[item["name"]] > item["max"]:
                    errors.append(f"Field {item['name']} must be at most {item['max']}")
            if item["type"] == "bool":
                if type(content[item["name"]]).__name__ != "bool":
                    errors.append(f"Field {item['name']} must be a boolean")
            if item["type"] == "list":
                if type(content[item["name"]]).__name__ != "list":
                    errors.append(f"Field {item['name']} must be a list")
                if "min" in item and len(content[item["name"]]) < item["min"]:
                    errors.append(f"Field {item['name']} must have at least {item['min']} items")
                if "max" in item and len(content[item["name"]]) > item["max"]:
                    errors.append(f"Field {item['name']} must have at most {item['max']} items")
                if "subtype" in item:
                    for i in range(len(content[item["name"]])):
                        if item["subtype"] == "int":
                            if type(content[item["name"]][i]).__name__ != "int":
                                errors.append(f"Field {item['name']} must be a list of integers")
                        if item["subtype"] == "str":
                            if type(content[item["name"]][i]).__name__ != "str":
                                errors.append(f"Field {item['name']} must be a list of strings")

    for key in content:
        if key not in [item["name"] for item in schema]:
            errors.append(f"Field {key} is not allowed here")
    return len(errors) == 0, errors