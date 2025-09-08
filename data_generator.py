#!/usr/bin/env python3
"""
Coherent Business Events Generator v4.0
Generates realistic, coherent, and configurable collections of canonical business events.
Can write events to a local JSON file or load them directly into Microsoft Fabric.
"""

import json
import uuid
import hashlib
import random
import argparse
import logging
import os
from datetime import datetime, timezone, timedelta
from faker import Faker

# --- Library Imports ---
try:
    import pandas as pd
    from deltalake import write_deltalake
except ImportError:
    print("Pandas or Deltalake libraries not found. Please install them to use the --load-to-fabric feature.")
    print("pip install pandas deltalake-v1.1.4 azure-identity")


# --- Basic Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
fake = Faker()


# --- Fabric Loading Function ---

def write_events_to_fabric(events: list, workspace_id: str, lakehouse_id: str):
    """
    Groups events by entity type and writes them to corresponding Delta tables in a Fabric Lakehouse.

    Authentication is handled via environment variables for a Service Principal.
    Please set the following environment variables before running:
    - AZURE_TENANT_ID
    - AZURE_CLIENT_ID
    - AZURE_CLIENT_SECRET
    """
    logger.info(f"Starting data load to Microsoft Fabric Lakehouse {lakehouse_id} in workspace {workspace_id}.")

    # Group events by their entity type
    events_by_type = {}
    for event in events:
        entity_type = event.get('entity_type')
        if not entity_type:
            continue
        if entity_type not in events_by_type:
            events_by_type[entity_type] = []
        events_by_type[entity_type].append(event)

    # For each entity type, create a DataFrame and write to a Delta table
    for entity_type, event_list in events_by_type.items():
        table_name = f"events_{entity_type}"
        logger.info(f"Preparing to write {len(event_list)} events to bronze table '{table_name}'.")

        # To handle nested objects, we serialize the payload into a JSON string.
        # This is a common pattern for bronze layers.
        writable_events = []
        for event in event_list:
            flat_event = event.copy()
            if 'payload' in flat_event and isinstance(flat_event['payload'], dict):
                flat_event['payload'] = json.dumps(flat_event['payload'])
            if 'manifest' in flat_event and isinstance(flat_event['manifest'], dict):
                flat_event['manifest'] = json.dumps(flat_event['manifest'])
            writable_events.append(flat_event)

        df = pd.DataFrame(writable_events)

        # Construct the ABFSS path for the Fabric Lakehouse table
        table_path = f"abfss://{workspace_id}@onelake.dfs.fabric.microsoft.com/{lakehouse_id}/Tables/{table_name}"

        logger.info(f"Writing DataFrame to Delta table at: {table_path}")
        try:
            write_deltalake(
                table_or_uri=table_path,
                data=df,
                mode="append",
                schema_mode="merge"
            )
            logger.info(f"Successfully appended {len(df)} records to '{table_name}'.")
        except Exception as e:
            logger.error(f"Failed to write to Fabric table '{table_name}'. Error: {e}")
            logger.error("Please ensure the Service Principal environment variables (AZURE_TENANT_ID, AZURE_CLIENT_ID, AZURE_CLIENT_SECRET) are set correctly and that the principal has 'Storage Blob Data Contributor' permissions on the Lakehouse.")
            return # Stop processing if one table fails

# --- Data Generation Logic ---

class BusinessEventGenerator:
    """Main generator for coherent business event collections."""

    def __init__(self):
        # ... (rest of the class is identical to the previous version)
        self.generated_entities = {
            'companies': {}, 'plants': {}, 'vendors': {}, 'customers': {},
            'materials': {}, 'addresses': {}, 'emails': {},
            'purchase_orders': {}, 'sales_orders': {}, 'stock_transfer_orders': {}
        }
        self.event_sequence = 0
        self.base_timestamp = datetime.now(timezone.utc) - timedelta(days=90)
        self.ref_data = {
            'order_types': ['NB', 'UB', 'ZNB', 'KB', 'LP'],
            'movement_types': ['101', '102', '161', '261', '301'],
            'incoterms': ['FOB', 'CIF', 'EXW', 'DDP']
        }

    def _get_sequential_timestamp(self) -> datetime:
        self.event_sequence += 1
        return self.base_timestamp + timedelta(minutes=self.event_sequence)

    def _generate_hash(self, content: dict) -> str:
        return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()

    def _create_canonical_event(self, entity_type: str, event_type: str, payload: dict) -> dict:
        ts_event = self._get_sequential_timestamp()
        header_data = {
            "event_type": event_type, "entity_type": entity_type, "schema_version": "1.0",
            "source_system": "SAP_S4HANA_2023", "sysid": "S4H", "mandt": "100",
            "ts_event": ts_event.isoformat(), "idempotency_key": str(uuid.uuid4()),
        }
        return {
            **header_data, "event_id": str(uuid.uuid4()), "correlation_id": str(uuid.uuid4()),
            "ts_ingest": (ts_event + timedelta(seconds=random.randint(1, 60))).isoformat(),
            "line_scope": "full", "lines_removed": [], "payload": payload,
            "header_hash": self._generate_hash(header_data),
            "manifest": {"generator": "CoherentEventGenerator/v4.0"}
        }

    def _generate_qty(self):
        return {"value": f"{random.uniform(1, 1000):.2f}", "uom": random.choice(['EA', 'KG', 'L', 'M'])}

    def _generate_incoterms(self):
        return {"code": random.choice(self.ref_data['incoterms']), "location": fake.city()}

    def generate_vendor(self):
        vendor_id = str(fake.random_number(digits=6, fix_len=True))
        payload = {"header": {
                "vendor_id": vendor_id, "name": fake.company(), "country": fake.country_code(),
                "incoterms": self._generate_incoterms(),
                "address": {"street": fake.street_address(), "city": fake.city(), "country": fake.country_code()},
                "emails": [{"email": fake.email()}],
                "changed_at": self._get_sequential_timestamp().isoformat()
            }, "record_hash": self._generate_hash({"id": vendor_id})
        }
        event = self._create_canonical_event('vendor', 'created', payload)
        self.generated_entities['vendors'][vendor_id] = payload
        return event

    def generate_material(self):
        material_id = str(fake.random_number(digits=8, fix_len=True))
        payload = {"header": {
                "material_id": material_id, "name": f"{fake.color_name().capitalize()} {fake.word()}",
                "country": fake.country_code(), "changed_at": self._get_sequential_timestamp().isoformat()
            }, "record_hash": self._generate_hash({"id": material_id})
        }
        event = self._create_canonical_event('material', 'created', payload)
        self.generated_entities['materials'][material_id] = payload
        return event

    def _generate_order_lines(self, plant_id, order_id):
        lines = []
        for i in range(random.randint(1, 5)):
            line_id = str((i + 1) * 10)
            material_id = random.choice(list(self.generated_entities['materials'].keys()))
            lines.append({
                "line_id": line_id, "item_no": line_id, "material_id": material_id, "plant_id": plant_id,
                "quantity": self._generate_qty(), "weight_net": self._generate_qty(),
                "volume": self._generate_qty(), "hs_code": str(fake.random_number(digits=6)),
                "changed_at": self._get_sequential_timestamp().isoformat(),
                "line_hash": self._generate_hash({"id": f"{order_id}-{line_id}"})
            })
        return lines

    def generate_purchase_order(self, is_sto=False):
        entity_type = 'sto' if is_sto else 'po'
        order_key = 'stock_transfer_orders' if is_sto else 'purchase_orders'
        vendor_id = random.choice(list(self.generated_entities['vendors'].keys()))
        plant_id = fake.bothify(text='PL##')
        company_code = fake.bothify(text='C###')
        po_number = fake.bothify(text='PO#######')
        lines = self._generate_order_lines(plant_id, po_number)
        payload = {"header": {
                "po_number": po_number, "company_code": company_code,
                "order_type": "UB" if is_sto else random.choice(self.ref_data['order_types']),
                "vendor_id": vendor_id, "is_sto": is_sto,
                "document_date": self._get_sequential_timestamp().isoformat(),
                "created_at": self._get_sequential_timestamp().isoformat(),
                "created_by": fake.user_name()
            }, "lines": lines
        }
        event = self._create_canonical_event(entity_type, 'created', payload)
        self.generated_entities[order_key][po_number] = payload
        return event

    def generate_sales_order(self):
        customer_id = random.choice(list(self.generated_entities['vendors'].keys()))
        plant_id = fake.bothify(text='PL##')
        company_code = fake.bothify(text='C###')
        so_number = fake.bothify(text='SO#######')
        lines = self._generate_order_lines(plant_id, so_number)
        payload = {"header": {
                "po_number": so_number, "company_code": company_code, "order_type": "OR",
                "vendor_id": customer_id,
                "created_at": self._get_sequential_timestamp().isoformat(),
                "created_by": fake.user_name()
            }, "lines": lines
        }
        event = self._create_canonical_event('so', 'created', payload)
        self.generated_entities['sales_orders'][so_number] = payload
        return event

    def generate_goods_receipt(self):
        if not self.generated_entities['purchase_orders']: return None
        po_number = random.choice(list(self.generated_entities['purchase_orders'].keys()))
        po_payload = self.generated_entities['purchase_orders'][po_number]
        lines = []
        for po_line in po_payload['lines']:
            received_qty_val = float(po_line['quantity']['value']) * random.uniform(0.8, 1.0)
            lines.append({
                "line_id": str(uuid.uuid4()), "po_item_no": po_line['item_no'],
                "quantity": {"value": f"{received_qty_val:.2f}", "uom": po_line['quantity']['uom']},
                "movement_type": random.choice(self.ref_data['movement_types']),
                "line_hash": self._generate_hash({"id": f"gr-{po_number}-{po_line['item_no']}"})
            })
        payload = {"header": {
                "gr_reference": {"po_number": po_number},
                "posting_date": self._get_sequential_timestamp().isoformat(),
                "plant_id": po_payload['lines'][0]['plant_id'],
                "created_at": self._get_sequential_timestamp().isoformat()
            }, "lines": lines
        }
        return self._create_canonical_event('gr', 'created', payload)

    def generate_coherent_collection(self, config: dict) -> list:
        events = []
        logger.info("Starting coherent event generation...")
        for _ in range(config.get('vendors')): events.append(self.generate_vendor())
        for _ in range(config.get('materials')): events.append(self.generate_material())
        logger.info(f"Generated {len(events)} master data events.")
        for _ in range(config.get('purchase_orders')): events.append(self.generate_purchase_order(is_sto=False))
        for _ in range(config.get('stock_transfer_orders')): events.append(self.generate_purchase_order(is_sto=True))
        for _ in range(config.get('sales_orders')): events.append(self.generate_sales_order())
        gr_count = min(config.get('goods_receipts'), len(self.generated_entities['purchase_orders']))
        for _ in range(gr_count):
            gr_event = self.generate_goods_receipt()
            if gr_event: events.append(gr_event)
        logger.info(f"Generated {len(events) - config.get('vendors') - config.get('materials')} transactional events.")
        update_count = int(len(events) * 0.1)
        for _ in range(update_count):
            if not events: break
            original_event = random.choice(events)
            update_event = dict(original_event)
            update_event['event_type'] = 'updated'
            update_event['event_id'] = str(uuid.uuid4())
            update_event['ts_event'] = self._get_sequential_timestamp().isoformat()
            events.append(update_event)
        logger.info(f"Generated {update_count} update events.")
        logger.info(f"Total events generated: {len(events)}")
        return events

def main():
    parser = argparse.ArgumentParser(description='Generate coherent business event collections.')
    # Generation config
    parser.add_argument('--vendors', type=int, default=10)
    parser.add_argument('--materials', type=int, default=20)
    parser.add_argument('--purchase-orders', type=int, default=15)
    parser.add_argument('--sales-orders', type=int, default=12)
    parser.add_argument('--stock-transfer-orders', type=int, default=5)
    parser.add_argument('--goods-receipts', type=int, default=8)
    parser.add_argument('--seed', type=int, help='Random seed for reproducible generation.')

    # Output config
    parser.add_argument('--output', '-o', default='coherent_events.json', help='Local output file path.')
    parser.add_argument('--no-file', action='store_true', help='Do not write events to a local file.')
    parser.add_argument('--pretty', action='store_true', help='Pretty print local JSON output.')

    # Fabric config
    parser.add_argument('--load-to-fabric', action='store_true', help='Enable loading data into Microsoft Fabric.')
    parser.add_argument('--fabric-workspace-id', type=str, help='Microsoft Fabric Workspace ID.')
    parser.add_argument('--fabric-lakehouse-id', type=str, help='Microsoft Fabric Lakehouse ID.')

    args = parser.parse_args()

    if args.seed:
        random.seed(args.seed)
        Faker.seed(args.seed)
        logger.info(f"Using random seed: {args.seed}")

    config = vars(args)

    generator = BusinessEventGenerator()
    events = generator.generate_coherent_collection(config)

    # --- Output to local file ---
    if not args.no_file:
        try:
            with open(args.output, 'w') as f:
                indent = 2 if args.pretty else None
                json.dump(events, f, indent=indent)
            logger.info(f"Successfully saved {len(events)} events to {args.output}")
        except Exception as e:
            logger.error(f"Failed to write to local file {args.output}. Error: {e}")

    # --- Output to Microsoft Fabric ---
    if args.load_to_fabric:
        if not all([args.fabric_workspace_id, args.fabric_lakehouse_id]):
            logger.error("To load to Fabric, you must provide --fabric-workspace-id and --fabric-lakehouse-id.")
        else:
            if 'pd' not in globals():
                 logger.error("Pandas/Deltalake libraries not loaded. Cannot write to Fabric.")
            else:
                write_events_to_fabric(events, args.fabric_workspace_id, args.fabric_lakehouse_id)

if __name__ == "__main__":
    main()
