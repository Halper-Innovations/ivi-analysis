from __future__ import annotations

import re


COMPANYFACTS_ALIAS_MAP: dict[str, list[str]] = {
    "cash": ["cash"],
    "cashandcashequivalents": ["cash"],
    "cashandcashequivalentsatcarryingvalue": ["cash"],
    "earningspersharebasic": ["net_income", "shares_outstanding"],
    "epsbasic": ["net_income", "shares_outstanding"],
    "revenues": ["revenue"],
    "revenue": ["revenue"],
    "salesrevenuegoodsnet": ["revenue"],
    "netincomeloss": ["net_income"],
    "netincome": ["net_income"],
    "operatingincomeloss": ["operating_income"],
    "operatingincome": ["operating_income"],
    "stockholdersequity": ["equity"],
    "shareholdersequity": ["equity"],
    "stockholdersequityincludingportionattributabletononcontrollinginterest": ["equity"],
    "stockholdersequityincludingportionattributabletononcontrollinginterestincludingtemporaryequity": ["equity"],
    "commonstocksharesoutstanding": ["shares_outstanding"],
    "entitycommonstocksharesoutstanding": ["shares_outstanding"],
    "weightedaveragenumberofsharesoutstandingbasic": ["shares_outstanding"],
    "weightedaveragenumberofsharesoutstandingdiluted": ["shares_outstanding"],
    "weightedaveragedilutedshares": ["shares_outstanding"],
    "weightedaveragesharesdiluted": ["shares_outstanding"],
    "dilutedshares": ["shares_outstanding"],
    "netcashprovidedbyusedinoperatingactivities": ["cfo"],
    "netcashprovidedbyusedinoperatingactivitiescontinuingoperations": ["cfo"],
    "operatingcashflow": ["cfo"],
    "cashflowfromoperations": ["cfo"],
    "operatingcashflows": ["cfo"],
    "freecashflow": ["cfo", "capex"],
    "fcf": ["cfo", "capex"],
    "paymentsfortpropertyplantandequipment": ["capex"],
    "paymentsforpropertyplantandequipment": ["capex"],
    "capitalexpenditures": ["capex"],
    "longtermdebt": ["total_debt"],
    "shorttermborrowings": ["total_debt"],
    "debt": ["total_debt"],
    "stockbasedcompensation": ["sbc"],
    "sharebasedcompensation": ["sbc"],
    "sbc": ["sbc"],
}


def companyfacts_alias_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value or "").lower())


def normalize_companyfacts_line_items(values: list[str]) -> tuple[list[str], list[str]]:
    normalized: list[str] = []
    translated: list[str] = []
    for value in values:
        key = companyfacts_alias_key(value)
        aliases = COMPANYFACTS_ALIAS_MAP.get(key)
        if aliases:
            normalized.extend(aliases)
            if aliases != [value]:
                translated.append(value)
        else:
            normalized.append(value)
    return list(dict.fromkeys(normalized)), list(dict.fromkeys(translated))
