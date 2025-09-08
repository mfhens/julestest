import json
import random
import uuid
import hashlib
from datetime import datetime, timezone, timedelta
from faker import Faker

fake = Faker()

# --- Reusable Component Generators ---

def generate_qty():
    """Generates a quantity object with a random value and unit of measure."""
    return {
        "value": f"{random.uniform(1, 1000):.2f}",
        "uom": random.choice(['EA', 'KG', 'L', 'M', 'TON'])
    }

def generate_incoterms():
    """Generates an incoterms object."""
    return {
        "code": random.choice(['FOB', 'CIF', 'EXW', 'DDP']),
        "location": fake.city()
    }

def generate_address_payload_data():
    """Generates a dictionary of address data."""
    return {
        "street": fake.street_address(),
        "city": fake.city(),
        "state": fake.state_abbr(),
        "zip_code": fake.zipcode(),
        "country": fake.country_code()
    }

# --- Payload Generators for each Entity Type ---

def generate_po_payload(is_sto=False):
    """Generates a realistic Purchase Order (PO) payload."""
    header = {
        "po_number": fake.bothify(text='PO#######'),
        "company_code": fake.bothify(text='C###'),
        "purchasing_group": fake.bothify(text='PG#'),
        "order_type": "STO" if is_sto else "NB",
        "vendor_id": str(fake.random_number(digits=6)),
        "incoterms": generate_incoterms(),
        "document_date": fake.iso8601(),
        "created_at": fake.iso8601(),
        "changed_at": fake.iso8601(),
        "created_by": fake.user_name(),
        "ship_from_partner": {
            "vendor_id": str(fake.random_number(digits=6))
        },
        "is_sto": is_sto
    }
    lines = []
    for i in range(random.randint(1, 5)):
        line_id = str((i + 1) * 10)
        line = {
            "line_id": line_id,
            "item_no": line_id,
            "material_id": str(fake.random_number(digits=8)),
            "plant_id": fake.bothify(text='PL##'),
            "quantity": generate_qty(),
            "weight_net": generate_qty(),
            "volume": generate_qty(),
            "hs_code": str(fake.random_number(digits=6)),
            "changed_at": fake.iso8601(),
            "line_hash": hashlib.sha256(str(uuid.uuid4()).encode()).hexdigest()
        }
        lines.append(line)

    return {"header": header, "lines": lines}

def generate_so_payload():
    """Generates a Sales Order (SO) payload, leveraging the PO structure."""
    so_payload = generate_po_payload()
    so_payload['header']['po_number'] = fake.bothify(text='SO#######')
    so_payload['header']['order_type'] = 'OR'
    # In a real SO, we'd have a customer_id instead of vendor_id,
    # but we are adhering to the provided schema which reuses the PO definition.
    return so_payload

def generate_sto_payload():
    """Generates a Stock Transport Order (STO) payload."""
    return generate_po_payload(is_sto=True)

def generate_gr_payload():
    """Generates a Goods Receipt (GR) payload."""
    header = {
        "gr_reference": {
            "po_number": fake.bothify(text='PO#######')
        },
        "posting_date": fake.iso8601(),
        "plant_id": fake.bothify(text='PL##'),
        "created_at": fake.iso8601(),
        "changed_at": fake.iso8601()
    }
    lines = []
    for i in range(random.randint(1, 3)):
        line = {
            "line_id": str((i + 1) * 10),
            "po_item_no": str((i + 1) * 10),
            "quantity": generate_qty(),
            "movement_type": "101",
            "batch": fake.bothify(text='BATCH-###???'),
            "storage_location": fake.bothify(text='SL##'),
            "line_hash": hashlib.sha256(str(uuid.uuid4()).encode()).hexdigest()
        }
        lines.append(line)
    return {"header": header, "lines": lines}

def generate_master_data_payload(entity_type):
    """
    Generates a master data payload (customer, vendor, etc.).
    The schema for these is identical, so we generate specific data based on type.
    Note: The schema specifies 'customer_id' for all these types, so we use
    that as the key, even though the logical ID might be a vendor_id, etc.
    """
    name = ""
    if entity_type == 'customer':
        name = fake.company()
    elif entity_type == 'vendor':
        name = fake.company() + " " + fake.company_suffix()
    elif entity_type == 'material':
        name = f"{fake.color_name().capitalize()} {fake.word().capitalize()}"
    elif entity_type == 'company':
        name = fake.company()
    elif entity_type == 'plant':
        name = f"{fake.city()} Plant"
    elif entity_type == 'address':
        name = f"Address for {fake.company()}"
    elif entity_type == 'email':
        name = fake.name()

    header = {
        "customer_id": str(fake.random_number(digits=7)),
        "name": name,
        "country": fake.country_code(),
        "changed_at": fake.iso8601()
    }

    # Add more specific fields based on entity type
    if entity_type in ['customer', 'vendor']:
        header["trading_partner"] = str(fake.random_number(digits=6))
        header["incoterms"] = generate_incoterms()
        header["address"] = generate_address_payload_data()
        header["emails"] = [{"email": fake.email()} for _ in range(random.randint(1, 2))]
    elif entity_type == 'address':
         header["address"] = generate_address_payload_data()
    elif entity_type == 'email':
        header["emails"] = [{"email": fake.email()}]

    return {
        "header": header,
        "record_hash": hashlib.sha256(str(uuid.uuid4()).encode()).hexdigest()
    }

# --- Main Event Generator ---

PAYLOAD_GENERATORS = {
    "po": generate_po_payload,
    "so": generate_so_payload,
    "sto": generate_sto_payload,
    "gr": generate_gr_payload,
    "customer": lambda: generate_master_data_payload("customer"),
    "vendor": lambda: generate_master_data_payload("vendor"),
    "material": lambda: generate_master_data_payload("material"),
    "company": lambda: generate_master_data_payload("company"),
    "plant": lambda: generate_master_data_payload("plant"),
    "address": lambda: generate_master_data_payload("address"),
    "email": lambda: generate_master_data_payload("email"),
}

def generate_event(entity_type):
    """
    Creates a comprehensive, realistic canonical business event.
    """
    if entity_type not in PAYLOAD_GENERATORS:
        raise ValueError(f"Unknown entity_type: {entity_type}")

    ts_event = datetime.now(timezone.utc)
    ts_ingest = ts_event + timedelta(seconds=random.randint(1, 60))

    payload = PAYLOAD_GENERATORS[entity_type]()

    header_data = {
        "event_type": random.choice(['created', 'updated']),
        "entity_type": entity_type,
        "schema_version": "1.0",
        "source_system": "SAP_S4HANA_2023",
        "sysid": "S4H",
        "mandt": "100",
        "ts_event": ts_event.isoformat(),
        "idempotency_key": str(uuid.uuid4()),
    }

    # Create a hash of the header for the header_hash field
    header_for_hash = json.dumps(header_data, sort_keys=True).encode()
    header_hash = hashlib.sha256(header_for_hash).hexdigest()

    event = {
        **header_data,
        "event_id": str(uuid.uuid4()),
        "correlation_id": str(uuid.uuid4()),
        "ts_ingest": ts_ingest.isoformat(),
        "line_scope": "full",
        "lines_removed": [],
        "payload": payload,
        "header_hash": header_hash,
        "manifest": {}
    }

    return event

# --- Main Execution Block ---

if __name__ == "__main__":
    entity_types = [
        "po", "so", "sto", "gr", "customer", "vendor",
        "material", "company", "plant", "address", "email"
    ]

    print("Generating one sample event for each entity type:")
    for entity_type in entity_types:
        print(f"\n--- Generating '{entity_type}' event ---")
        business_event = generate_event(entity_type)
        print(json.dumps(business_event, indent=2))
