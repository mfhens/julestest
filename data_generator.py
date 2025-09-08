#!/usr/bin/env python3
"""
Coherent Business Events Generator v3.0
Generates realistic, coherent, and configurable collections of canonical business events.
"""

import json
import uuid
import hashlib
import random
import argparse
import logging
from datetime import datetime, timezone, timedelta
from faker import Faker

# --- Basic Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
fake = Faker()

class BusinessEventGenerator:
    """Main generator for coherent business event collections."""

    def __init__(self):
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
            "manifest": {"generator": "CoherentEventGenerator/v3.0"}
        }

    def _generate_qty(self):
        return {"value": f"{random.uniform(1, 1000):.2f}", "uom": random.choice(['EA', 'KG', 'L', 'M'])}

    def _generate_incoterms(self):
        return {"code": random.choice(self.ref_data['incoterms']), "location": fake.city()}

    # --- Master Data Generators ---

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

    # --- Transactional Data Generators ---

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
        plant_id = fake.bothify(text='PL##') # Assume plants are just codes for now
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
        # For simplicity, re-using vendors as customers, but could be separate
        customer_id = random.choice(list(self.generated_entities['vendors'].keys()))
        plant_id = fake.bothify(text='PL##')
        company_code = fake.bothify(text='C###')
        so_number = fake.bothify(text='SO#######')

        lines = self._generate_order_lines(plant_id, so_number)

        payload = {"header": {
                # Adhering to schema where SO reuses PO structure
                "po_number": so_number, "company_code": company_code, "order_type": "OR",
                "vendor_id": customer_id, # Schema uses vendor_id for customer
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

    # --- Orchestrator ---

    def generate_coherent_collection(self, config: dict) -> list:
        events = []
        logger.info("Starting coherent event generation...")

        # Step 1: Generate Master Data
        for _ in range(config.get('vendors')): events.append(self.generate_vendor())
        for _ in range(config.get('materials')): events.append(self.generate_material())
        logger.info(f"Generated {len(events)} master data events.")

        # Step 2: Generate Transactional Data
        for _ in range(config.get('purchase_orders')): events.append(self.generate_purchase_order(is_sto=False))
        for _ in range(config.get('stock_transfer_orders')): events.append(self.generate_purchase_order(is_sto=True))
        for _ in range(config.get('sales_orders')): events.append(self.generate_sales_order())

        gr_count = min(config.get('goods_receipts'), len(self.generated_entities['purchase_orders']))
        for _ in range(gr_count):
            gr_event = self.generate_goods_receipt()
            if gr_event: events.append(gr_event)

        logger.info(f"Generated {len(events) - config.get('vendors') - config.get('materials')} transactional events.")

        # Step 3: Generate Update Events
        update_count = int(len(events) * 0.1) # 10% update events
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
    parser.add_argument('--output', '-o', default='coherent_events.json')
    parser.add_argument('--vendors', type=int, default=10)
    parser.add_argument('--materials', type=int, default=20)
    parser.add_argument('--purchase-orders', type=int, default=15)
    parser.add_argument('--sales-orders', type=int, default=12)
    parser.add_argument('--stock-transfer-orders', type=int, default=5)
    parser.add_argument('--goods-receipts', type=int, default=8)
    parser.add_argument('--pretty', action='store_true')
    parser.add_argument('--seed', type=int)

    args = parser.parse_args()

    if args.seed:
        random.seed(args.seed)
        Faker.seed(args.seed)
        logger.info(f"Using random seed: {args.seed}")

    config = vars(args)

    generator = BusinessEventGenerator()
    events = generator.generate_coherent_collection(config)

    with open(args.output, 'w') as f:
        indent = 2 if args.pretty else None
        json.dump(events, f, indent=indent)

    logger.info(f"Successfully saved {len(events)} events to {args.output}")

if __name__ == "__main__":
    main()
