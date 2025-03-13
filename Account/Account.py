class Account:
    canadian_accounts = ["A", "C", "E", "G", "L", "Q", "S", "T", "X", "Y"]
    us_accounts = ["B", "D", "F", "H", "M", "6", "R", "U"]

    managed_prefixes = ["067", "087", "077", "057", "047"]
    fee_based_prefixes = ["069", "089", "079", "029", "039", "059", "049"]

    registered_suffixes = ["I", "J", "K", "O", "P", "Q", "6", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
    registered_sub_types = ["I", "J", "L", "C", "P", "T", "N", "X", "R", "S", "G", "D", "V-IND", "E-FAM", "U"]
    
    empty = [None, "", " "]
    
    # Columns that cannot be empty
    # mandatory_columns = [funds, monitor_code, uci, uci_percentage, recipient_code]

    @staticmethod
    # Gets the column index using the exact name of the column
    def get_column_number_from_name(sheet, column_name: str) -> int or None:
        for i, column in enumerate(sheet.iter_cols()):
            if column[0].value == column_name:
                print(column_name, " ", i)
                return i
        print(column_name, " NONE")
        return None

    def __init__(self, sheet, row):
        self.sheet = sheet
        self.row = row
        # Variables to hold column indexes
        self.acct_number = self.row[Account.get_column_number_from_name(self.sheet, "Account Number")].value
        self.funds = self.row[Account.get_column_number_from_name(self.sheet, "Funds (Account)")].value
        self.port_type = self.row[Account.get_column_number_from_name(self.sheet, "Portfolio Type")].value
        self.min_commission = self.row[Account.get_column_number_from_name(self.sheet, "Minimum Commission?")].value
        self.type_com = self.row[Account.get_column_number_from_name(self.sheet, "Type of Commission")].value
        self.discretionary = self.row[Account.get_column_number_from_name(self.sheet, "Discretionary")].value
        self.credit_int = self.row[Account.get_column_number_from_name(self.sheet, "Credit Int")].value
        self.debit_int = self.row[Account.get_column_number_from_name(self.sheet, "Debit Int")].value
        self.acct_sub_type = self.row[Account.get_column_number_from_name(self.sheet, "Account Sub Type")].value
        self.acct_type = self.row[Account.get_column_number_from_name(self.sheet, "Account Type")].value
        self.open_date = self.row[Account.get_column_number_from_name(self.sheet, "Open Date (Account)")].value
        self.confirm_copies = self.row[Account.get_column_number_from_name(self.sheet, "Confirm Copies")].value
        self.e_conf = self.row[Account.get_column_number_from_name(self.sheet, "E-Confirm")].value
        self.last_user = self.row[Account.get_column_number_from_name(self.sheet, "Last User")].value
        self.monitor_code = self.row[Account.get_column_number_from_name(self.sheet, "Monitor Code")].value
        self.kyc_obj = self.row[Account.get_column_number_from_name(self.sheet, "KYC Objective Code")].value
        self.time_horizon = self.row[Account.get_column_number_from_name(self.sheet, "Time Horizon!Code")].value
        self.nrt_code = self.row[Account.get_column_number_from_name(self.sheet, "NRT Code")].value
        self.residence = self.row[Account.get_column_number_from_name(self.sheet, "Residence (Account)")].value
        self.recipient_code = self.row[Account.get_column_number_from_name(self.sheet, "1042 Recipient Code")].value
        self.crm2_flag = self.row[Account.get_column_number_from_name(self.sheet, "CRM2 Eligibility Flag")].value
        self.class_acct = self.row[Account.get_column_number_from_name(self.sheet, "Class (Account)")].value
        self.statement_copies = self.row[Account.get_column_number_from_name(self.sheet, "Statement Copies")].value
        self.send_statement = self.row[Account.get_column_number_from_name(self.sheet, "Send Statement ")].value
        self.np54 = self.row[Account.get_column_number_from_name(self.sheet, "NP54-101 ")].value
        self.ni54 = self.row[Account.get_column_number_from_name(self.sheet, "NI54-b ")].value
        self.lei = self.row[Account.get_column_number_from_name(self.sheet, "LEI/CICI")].value
        self.invest_knowledge = self.row[Account.get_column_number_from_name(self.sheet, "Invest Knowledge")].value
        self.uci = self.row[Account.get_column_number_from_name(self.sheet, "CDIC Unique Client Identifier")].value
        self.uci_percentage = self.row[Account.get_column_number_from_name(self.sheet, "UCI Percentage")].value
        self.recip_type = self.row[Account.get_column_number_from_name(self.sheet, "Recipient Type")].value
        self.title = self.row[Account.get_column_number_from_name(self.sheet, "Title")].value
        self.structured_addr = self.row[Account.get_column_number_from_name(self.sheet, "Structured Address?")].value
        self.addr_line = self.row[Account.get_column_number_from_name(self.sheet, "Addr Line 1")].value

        self.number_prefix = self.acct_number[:3]
        self.number_suffix = self.acct_number[-1]

        self.is_canadian = (self.number_suffix in Account.canadian_accounts)
        self.is_us = (self.number_suffix in Account.us_accounts)
        self.is_managed = (self.number_prefix in Account.managed_prefixes)
        self.is_fee_based = (self.number_prefix in Account.fee_based_prefixes)
        self.is_commission = not(self.is_managed or self.is_fee_based)

        self.is_registered = (self.number_suffix in Account.registered_suffixes)


