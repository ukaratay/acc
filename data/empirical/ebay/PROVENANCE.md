# eBay Auction Data Provenance

The CSV files in this directory are bid-level eBay auction datasets from the
data collection accompanying Wolfgang Jank and Galit Shmueli, *Modeling Online
Auctions* (Wiley-Blackwell, 2010, ISBN 978-0-470-47565-2), distributed by the
authors as a companion resource for research and teaching.

Contributor credits, preserved in the original directory names:

- **Xbox game consoles** — thanks to Shanshan Wang
- **Cartier wristwatches** — thanks to Sharad Borle
- **Palm Pilot M515 PDAs** — from the book's companion datasets

The files are redistributed here unmodified, with attribution, as the inputs
to the replication pipeline (`scripts/data/clean_ebay_data.py` consolidates
them into `ebay_auctions.parquet`). They contain auction prices, bid amounts,
timestamps, and anonymized bidder handles; no personally identifying
information beyond public eBay usernames at the time of collection.

If you reuse these data, cite Jank and Shmueli (2010) and credit the
contributors above.
