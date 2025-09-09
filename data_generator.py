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
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import TimestampType, StringType

# --- Basic Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
fake = Faker()

class BusinessEventGenerator:
    """Main generator for coherent business event collections."""

    def __init__(self):
        self.generated_entities = {
            'companies': {}, 'plants': {}, 'vendors': {}, 'customers': {}, 'materials': {},
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

    def _create_canonical_event(
        self, entity_type: str, event_type: str, payload: dict, header_hash: str, extract_id: str
    ) -> dict:
        """Creates a canonical event structure aligned with the target Delta table."""
        ts_event = self._get_sequential_timestamp()
        return {
            "idempotency_key": str(uuid.uuid4()),
            "entity_type": entity_type,
            "event_type": event_type,
            "sysid": "S4H",
            "mandt": "100",
            "header_hash": header_hash,
            "line_scope": "full",
            "lines_removed": "[]",  # Default to empty JSON array string
            "payload": payload,  # Will be serialized to string in the writer
            "ts_event": ts_event,
            "ts_ingest": ts_event + timedelta(seconds=random.randint(1, 60)),
            "extract_id": extract_id,
            "source_path": "data_generator.py",
            "schema_version": "v1",
        }

    def _generate_qty(self):
        return {"value": f"{random.uniform(1, 1000):.2f}", "uom": random.choice(['EA', 'KG', 'L', 'M'])}

    def _generate_incoterms(self):
        return {"code": random.choice(self.ref_data['incoterms']), "location": fake.city()}

    # --- Master Data Generators ---

    def generate_vendor(self, extract_id: str):
        vendor_id = str(fake.random_number(digits=6, fix_len=True))
        header_hash = self._generate_hash({"id": vendor_id})
        payload = {"header": {
                "vendor_id": vendor_id, "name": fake.company(), "country": fake.country_code(),
                "incoterms": self._generate_incoterms(),
                "address": {"street": fake.street_address(), "city": fake.city(), "country": fake.country_code()},
                "emails": [{"email": fake.email()}],
                "changed_at": self._get_sequential_timestamp().isoformat()
            }, "record_hash": header_hash
        }
        event = self._create_canonical_event('vendor', 'created', payload, header_hash, extract_id)
        self.generated_entities['vendors'][vendor_id] = payload
        return event

    def generate_material(self, extract_id: str):
        material_id = str(fake.random_number(digits=8, fix_len=True))
        header_hash = self._generate_hash({"id": material_id})
        payload = {"header": {
                "material_id": material_id, "name": f"{fake.color_name().capitalize()} {fake.word()}",
                "country": fake.country_code(), "changed_at": self._get_sequential_timestamp().isoformat()
            }, "record_hash": header_hash
        }
        event = self._create_canonical_event('material', 'created', payload, header_hash, extract_id)
        self.generated_entities['materials'][material_id] = payload
        return event

    def generate_company(self, extract_id: str):
        company_id = fake.bothify(text='C###')
        header_hash = self._generate_hash({"id": company_id})
        payload = {"header": {
                "company_id": company_id, "name": fake.company(), "country": fake.country_code(),
                "currency": fake.currency_code(),
                "changed_at": self._get_sequential_timestamp().isoformat()
            }, "record_hash": header_hash
        }
        event = self._create_canonical_event('company', 'created', payload, header_hash, extract_id)
        self.generated_entities['companies'][company_id] = payload
        return event

    def generate_plant(self, extract_id: str):
        plant_id = fake.bothify(text='PL##')
        header_hash = self._generate_hash({"id": plant_id})
        payload = {"header": {
                "plant_id": plant_id, "name": f"{fake.city()} Plant",
                "address": {"street": fake.street_address(), "city": fake.city(), "country": fake.country_code()},
                "changed_at": self._get_sequential_timestamp().isoformat()
            }, "record_hash": header_hash
        }
        event = self._create_canonical_event('plant', 'created', payload, header_hash, extract_id)
        self.generated_entities['plants'][plant_id] = payload
        return event

    def generate_customer(self, extract_id: str):
        customer_id = str(fake.random_number(digits=7, fix_len=True))
        header_hash = self._generate_hash({"id": customer_id})
        payload = {"header": {
                "customer_id": customer_id, "name": fake.name(), "country": fake.country_code(),
                "address": {"street": fake.street_address(), "city": fake.city(), "country": fake.country_code()},
                "emails": [{"email": fake.email()}],
                "changed_at": self._get_sequential_timestamp().isoformat()
            }, "record_hash": header_hash
        }
        event = self._create_canonical_event('customer', 'created', payload, header_hash, extract_id)
        self.generated_entities['customers'][customer_id] = payload
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

    def generate_purchase_order(self, extract_id: str, is_sto=False):
        entity_type = 'sto' if is_sto else 'po'
        order_key = 'stock_transfer_orders' if is_sto else 'purchase_orders'

        vendor_id = random.choice(list(self.generated_entities['vendors'].keys()))
        plant_id = random.choice(list(self.generated_entities['plants'].keys()))
        company_code = random.choice(list(self.generated_entities['companies'].keys()))
        po_number = fake.bothify(text='PO#######')
        header_hash = self._generate_hash({"id": po_number})

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
        event = self._create_canonical_event(entity_type, 'created', payload, header_hash, extract_id)
        self.generated_entities[order_key][po_number] = payload
        return event

    def generate_sales_order(self, extract_id: str):
        customer_id = random.choice(list(self.generated_entities['customers'].keys()))
        plant_id = random.choice(list(self.generated_entities['plants'].keys()))
        company_code = random.choice(list(self.generated_entities['companies'].keys()))
        so_number = fake.bothify(text='SO#######')
        header_hash = self._generate_hash({"id": so_number})

        lines = self._generate_order_lines(plant_id, so_number)

        payload = {"header": {
                # Adhering to schema where SO reuses PO structure
                "po_number": so_number, "company_code": company_code, "order_type": "OR",
                "vendor_id": customer_id, # Schema uses vendor_id for customer
                "created_at": self._get_sequential_timestamp().isoformat(),
                "created_by": fake.user_name()
            }, "lines": lines
        }
        event = self._create_canonical_event('so', 'created', payload, header_hash, extract_id)
        self.generated_entities['sales_orders'][so_number] = payload
        return event

    def generate_goods_receipt(self, extract_id: str):
        if not self.generated_entities['purchase_orders']: return None
        po_number = random.choice(list(self.generated_entities['purchase_orders'].keys()))
        po_payload = self.generated_entities['purchase_orders'][po_number]
        gr_doc_id = f"GR-{po_number}-{str(uuid.uuid4())[:8]}"
        header_hash = self._generate_hash({"id": gr_doc_id})

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
                "gr_document_id": gr_doc_id,
                "gr_reference": {"po_number": po_number},
                "posting_date": self._get_sequential_timestamp().isoformat(),
                "plant_id": po_payload['lines'][0]['plant_id'],
                "created_at": self._get_sequential_timestamp().isoformat()
            }, "lines": lines
        }
        return self._create_canonical_event('gr', 'created', payload, header_hash, extract_id)

    # --- Orchestrator ---

    def generate_coherent_collection(self, config: dict) -> list:
        events = []
        extract_id = str(uuid.uuid4())
        logger.info(f"Starting coherent event generation for extract_id: {extract_id}")

        # Step 1: Generate Master Data
        for _ in range(config.get('companies')): events.append(self.generate_company(extract_id))
        for _ in range(config.get('plants')): events.append(self.generate_plant(extract_id))
        for _ in range(config.get('vendors')): events.append(self.generate_vendor(extract_id))
        for _ in range(config.get('customers')): events.append(self.generate_customer(extract_id))
        for _ in range(config.get('materials')): events.append(self.generate_material(extract_id))
        logger.info(f"Generated {len(events)} master data events.")

        # Step 2: Generate Transactional Data
        if not all(len(self.generated_entities[k]) > 0 for k in ['vendors', 'materials', 'plants', 'companies', 'customers']):
            logger.warning("Skipping transactional data generation due to missing master data.")
        else:
            for _ in range(config.get('purchase_orders')): events.append(self.generate_purchase_order(extract_id, is_sto=False))
            for _ in range(config.get('stock_transfer_orders')): events.append(self.generate_purchase_order(extract_id, is_sto=True))
            for _ in range(config.get('sales_orders')): events.append(self.generate_sales_order(extract_id))

            gr_count = min(config.get('goods_receipts'), len(self.generated_entities['purchase_orders']))
            for _ in range(gr_count):
                gr_event = self.generate_goods_receipt(extract_id)
                if gr_event: events.append(gr_event)

            logger.info(f"Generated transactional events.")

        # Step 3: Generate Update Events
        update_count = int(len(events) * 0.1) # 10% update events
        for _ in range(update_count):
            if not events: break
            original_event = random.choice(events)

            update_event = dict(original_event)
            update_event['event_type'] = 'updated'
            update_event['idempotency_key'] = str(uuid.uuid4())
            update_event['ts_event'] = self._get_sequential_timestamp()

            # Ensure payload is a dictionary before modification
            if isinstance(update_event['payload'], str):
                try:
                    update_event['payload'] = json.loads(update_event['payload'])
                except json.JSONDecodeError:
                    logger.warning("Could not decode payload string for update event, skipping modification.")
                    continue

            if 'header' in update_event.get('payload', {}):
                 update_event['payload']['header']['changed_at'] = update_event['ts_event'].isoformat()

            events.append(update_event)
        logger.info(f"Generated {update_count} update events.")

        logger.info(f"Total events generated: {len(events)}")
        return events

def save_events_to_fabric(events: list, lakehouse_name: str):
    """
    Saves a list of generated events to their corresponding Delta tables in Microsoft Fabric.
    """
    if not events:
        logger.info("No events to save.")
        return

    logger.info("Initializing Spark session for Fabric.")
    try:
        spark = SparkSession.builder.appName("CoherentEventsFabricWriter").getOrCreate()
    except Exception as e:
        logger.error(f"Failed to create Spark session: {e}")
        return

    # Group events by their entity type
    events_by_type = {}
    for event in events:
        entity_type = event.get('entity_type')
        # Handle STO mapping to PO table
        key = 'po' if entity_type == 'sto' else entity_type
        if key not in events_by_type:
            events_by_type[key] = []
        events_by_type[key].append(event)

    logger.info(f"Found event types to process: {list(events_by_type.keys())}")

    entity_to_table_map = {
        'po': 'events_po',
        'so': 'events_so',
        'gr': 'events_gr',
        'delivery': 'events_delivery',
        'vendor': 'events_vendor',
        'customer': 'events_customer',
        'material': 'events_material',
        'plant': 'events_plant',
        'company': 'events_company',
    }

    for entity_type, batch in events_by_type.items():
        if entity_type not in entity_to_table_map:
            logger.warning(f"No table mapping found for entity_type '{entity_type}'. Skipping.")
            continue

        table_name = f"{lakehouse_name}.bronze.{entity_to_table_map[entity_type]}"
        logger.info(f"Processing {len(batch)} events for entity '{entity_type}' into table '{table_name}'")

        # Serialize payload dict to JSON string
        for event in batch:
            if isinstance(event.get('payload'), dict):
                event['payload'] = json.dumps(event['payload'])

        try:
            df = spark.createDataFrame(batch)

            df = df.withColumn("ts_event", col("ts_event").cast(TimestampType())) \
                   .withColumn("ts_ingest", col("ts_ingest").cast(TimestampType())) \
                   .withColumn("payload", col("payload").cast(StringType()))

            final_cols = [
                'idempotency_key', 'entity_type', 'event_type', 'sysid', 'mandt',
                'header_hash', 'line_scope', 'lines_removed', 'payload', 'ts_event',
                'ts_ingest', 'extract_id', 'source_path', 'schema_version'
            ]
            df = df.select(final_cols)

            logger.info(f"Writing DataFrame to Delta table: {table_name}")
            df.write.format("delta").mode("append").option("mergeSchema", "true").saveAsTable(table_name)
            logger.info(f"Successfully wrote {df.count()} records to {table_name}")

        except Exception as e:
            logger.error(f"Failed to write to table {table_name}: {e}")
            # If running locally or in debug, it's useful to see the schema
            if 'spark' in locals():
                df.printSchema()

def main():
    parser = argparse.ArgumentParser(description='Generate coherent business event collections.')
    # Output options
    parser.add_argument('--output-format', type=str, default='json', choices=['json', 'fabric'], help='The output format for the generated events.')
    parser.add_argument('--lakehouse-name', type=str, default='lh_cec_bronze', help='The name of the Fabric lakehouse (used when output-format is fabric).')
    parser.add_argument('--output', '-o', default='coherent_events.json', help='Output file name (used when output-format is json).')

    # Entity count options
    parser.add_argument('--companies', type=int, default=5, help='Number of companies to generate.')
    parser.add_argument('--plants', type=int, default=10, help='Number of plants to generate.')
    parser.add_argument('--vendors', type=int, default=10, help='Number of vendors to generate.')
    parser.add_argument('--customers', type=int, default=20, help='Number of customers to generate.')
    parser.add_argument('--materials', type=int, default=50, help='Number of materials to generate.')
    parser.add_argument('--purchase-orders', type=int, default=15, help='Number of purchase orders to generate.')
    parser.add_argument('--sales-orders', type=int, default=12, help='Number of sales orders to generate.')
    parser.add_argument('--stock-transfer-orders', type=int, default=5, help='Number of stock transfer orders to generate.')
    parser.add_argument('--goods-receipts', type=int, default=8, help='Number of goods receipts to generate.')
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

    if args.output_format == 'json':
        with open(args.output, 'w') as f:
            indent = 2 if args.pretty else None
            json.dump(events, f, indent=indent, default=str)
        logger.info(f"Successfully saved {len(events)} events to {args.output}")
    elif args.output_format == 'fabric':
        if not args.lakehouse_name:
            logger.error("A lakehouse name must be provided for the 'fabric' output format.")
            return
        save_events_to_fabric(events, args.lakehouse_name)
    else:
        logger.error(f"Unsupported output format: {args.output_format}")

if __name__ == "__main__":
    main()
