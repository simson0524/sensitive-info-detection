# scripts/run_export_table.py

from src.database.connection import db_manager
from src.database import crud, models
import os

# --------------------------------------------------------------------------
# [CONFIG] 
# --------------------------------------------------------------------------
export_target_table = "DomainTermMatrix"
csv_root_dir = "/home/student1/sensitive-info-detection/data/db_storage"
csv_file_path = os.path.join(csv_root_dir, export_target_table)


try:
    with db_manager.get_db() as session:
        dtm_path = os.path.join(csv_root_dir, 'domain_term_matrix.csv')
        term_path = os.path.join(csv_root_dir, 'term.csv')
        domain_path = os.path.join(csv_root_dir, 'domain.csv')
        crud.export_table_to_csv(session, models.DomainTermMatrix, dtm_path)
        crud.export_table_to_csv(session, models.Term, term_path)
        crud.export_table_to_csv(session, models.Domain, domain_path)

except Exception as e:
    print(f"[Error] {e}")