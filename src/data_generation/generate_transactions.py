"""
Synthetic transaction failure data generator.

Generates cards + UPI transactions with method-conditional failure logic,
grounded in real public bank/NPCI response code taxonomies.

Two labels per row:
  - failure_category: hard_decline | soft_decline | customer_dropoff | fraud_block
  - retry_success: bool, whether a retry/nudge would have recovered the transaction
                   (always False for fraud_block by construction)

Design notes (see docs/decisions.md):
  - Cards skew toward hard_decline / fraud_block (issuer/network-driven failures)
  - UPI skews toward soft_decline / customer_dropoff (operational failures)
  - Deliberate overlap is introduced between soft_decline and customer_dropoff
    for a subset of UPI transactions (see docs/failure_stories.md, Candidate 1)

TODO:
  - Define CARD_ERROR_CODES and UPI_ERROR_CODES taxonomies
  - Implement generate_card_transaction()
  - Implement generate_upi_transaction()
  - Implement method-conditional failure_category assignment
  - Implement retry_success assignment conditioned on category + features
  - Write to data/raw/transactions.csv
"""

def main():
    raise NotImplementedError("Data generator not yet implemented")


if __name__ == "__main__":
    main()
