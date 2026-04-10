# -*- coding: utf-8 -*-
import os
import sys
from pathlib import Path

base_path = Path(__file__).resolve().parent
sys.path.insert(0, str(base_path / 'dataset'))
sys.path.insert(0, str(base_path.parent / 'src' / 'libs'))
sys.path = list(dict.fromkeys(sys.path))

import CNoXiDatabase
from log import getLogger


class CDataBaseSelection:
    """DB selection configuration"""

    def __init__(self):
        self._logger = getLogger("DB_selection")

    def get_db_by_name(self, db_name: str) -> None:
        """
        Get db

        Args:
            db_name: database name

        Raises:
            Exception: Stop the code if the db to select does not exist.
        """
        if db_name == "NoXi":
            result = CNoXiDatabase
        else:
            self._m_logger.LOG_ERROR(f"Selected db {db_name} is not available")
            raise Exception(f"Throwing error: Selected db {db_name} is not available")
