"""F3: count the orders in a long list that match a condition. The level sets the list's length (a 16K-token local
context cannot hold the longest); the hidden bin sets the condition: one field (low), two (mid), or two plus a
threshold, with near misses that look like matches (high)."""
import random

from purse.jobs import register_family, same_integer

__all__ = ["AGGREGATION", "RECORDS", "Aggregation"]

RECORDS = {1: 40, 2: 120, 3: 400, 4: 1000, 5: 2500}
REGIONS = ("North", "South", "East", "West", "Central")
NEAR_REGIONS = {"North": ("North-East", "Northwest"), "South": ("South-East", "Southwest")}
STATUSES = ("shipped", "pending", "cancelled", "delivered")
FIRST = ("Ana", "Ben", "Chen", "Dara", "Eli", "Femi", "Gus", "Hana", "Ivo", "Juno", "Kai", "Lena", "Mo", "Nia", "Oren",
         "Pia", "Quin", "Rosa", "Sami", "Tova")
LAST = ("Abe", "Brandt", "Costa", "Diaz", "Ekwueme", "Fox", "Gupta", "Haas", "Ito", "Jensen", "Kowal", "Lund", "Moreau",
        "Novak", "Okafor", "Park", "Quist", "Reyes", "Silva", "Tanaka")

Order = dict[str, object]


def _order(rng: random.Random, region: str, status: str, total: float) -> Order:
    return {"date": f"2026-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}",
            "customer": f"{rng.choice(FIRST)} {rng.choice(LAST)}", "region": region, "items": rng.randint(1, 9),
            "total": total, "status": status}


def _total(rng: random.Random) -> float:
    return rng.randint(500, 99_999) / 100


class Aggregation:
    name = "aggregation"
    checkable = False

    def generate(self, level: int, hidden: str, rng: random.Random) -> tuple[str, str, dict[str, int]]:
        n = RECORDS[level]
        region = rng.choice(tuple(NEAR_REGIONS) if hidden == "high" else REGIONS)
        status = "shipped" if hidden == "high" else rng.choice(STATUSES)
        threshold = rng.randint(200, 800) if hidden == "high" else None

        def matches(o: Order) -> bool:
            return (o["region"] == region and (hidden == "low" or o["status"] == status)
                    and (threshold is None or float(o["total"]) > threshold))  # type: ignore[arg-type]

        count = rng.randint(3, max(3, n // 5))
        orders = []
        for _ in range(count):
            total = rng.randint(threshold * 100 + 1, 99_999) / 100 if threshold else _total(rng)
            orders.append(_order(rng, region, status if hidden != "low" else rng.choice(STATUSES), total))
        if hidden == "high":  # at least one of each near miss, none of which counts
            assert threshold is not None
            for _ in range(max(1, n // 50)):
                orders.append(_order(rng, rng.choice(NEAR_REGIONS[region]), status, rng.randint(threshold * 100 + 1,
                                                                                                    99_999) / 100))
                orders.append(_order(rng, region, "shipped-late", rng.randint(threshold * 100 + 1, 99_999) / 100))
                orders.append(_order(rng, region, status, float(threshold)))
        regions = REGIONS + (NEAR_REGIONS[region] if hidden == "high" else ())
        statuses = STATUSES + (("shipped-late",) if hidden == "high" else ())
        while len(orders) < n:
            o = _order(rng, rng.choice(regions), rng.choice(statuses), _total(rng))
            if not matches(o):
                orders.append(o)
        rng.shuffle(orders)
        lines = [f"#{1000 + i} | {o['date']} | customer {o['customer']} | region {o['region']} | items {o['items']} | "
                 f"total ${float(o['total']):.2f} | status {o['status']}" for i, o in enumerate(orders)]  # type: ignore[arg-type]
        condition = f"are from region {region}"
        if hidden != "low":
            condition += f" and have status {status}"
        if threshold is not None:
            condition = f"are from region {region}, have status {status}, and have a total above ${threshold:.2f}"
        task = f"Here are {n} orders, one per line.\n\n" + "\n".join(lines) + f"\n\nHow many orders {condition}? " \
               "Answer with the count."
        return task, str(sum(matches(o) for o in orders)), {"records": n}

    def correct(self, answer: str, response: str) -> bool:
        return same_integer(answer, response)


AGGREGATION = register_family(Aggregation())
