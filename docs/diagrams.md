# Diagrams

How the harness fits together, drawn from the code (module imports and `harness/models.py`).
File-level detail is in the Components table in [ARCHITECTURE.md](../ARCHITECTURE.md).
GitHub renders these Mermaid diagrams; in VS Code, a Mermaid preview extension does.

## 1. Overview: the big picture

Start here. Each day, bars go in and trades come out. The only thing that differs
between a backtest and paper trading is the broker at the end.

```mermaid
flowchart LR
    data["1 Data<br/>daily bars from Alpaca"]
    strat["2 Strategy<br/>your code: bars in, Signals out"]
    cons["3 Construction<br/>which signals, how many shares"]
    risk["4 Risk<br/>hard limits: shrink or reject"]
    broker{"5 Broker"}
    sim["Simulated fills<br/>backtest"]
    alpaca["Alpaca paper account<br/>paper trading"]
    reports["6 Reports<br/>equity, trades, summary"]
    safety["Safety net<br/>journal, reconcile,<br/>kill switch, approvals"]

    data --> strat --> cons --> risk --> broker
    broker -- "backtest" --> sim
    broker -- "paper" --> alpaca
    sim --> reports
    alpaca --> reports
    safety -. "guards" .- alpaca
```

How the overview boxes map to the files in the detailed diagram below:

| Overview box | Files (section 2) |
|---|---|
| 1 Data | `config.py`, `keys.py`, `data.py`, `view.py` |
| 2 Strategy | `strategies/*.py`, `guard.py` |
| 3 Construction | `construction.py`, `costs.py` |
| 4 Risk | `risk.py` |
| 5 Broker | `engine.py` drives the day; `broker/base.py` is the slot; `broker/sim.py` (backtest) or `orders.py`, `broker/venue.py`, `broker/alpaca.py` (paper) fill it |
| 6 Reports | `report.py` (backtest), `paper_report.py` (paper) |
| Safety net | `journal.py`, `positions.py`, `killswitch.py`, `inbox.py`, `alerts.py`, `profiles.py` |
| Not shown | Who starts things: `run.py` (backtest CLI), `paper.py` (paper runner), and the operator's `cli.py` with `service.py` (views) and `control.py` (commands) |

Boxes 1 to 4 and the engine are the same code in a backtest and in paper trading. That
is the "one code path" rule, and it is why a backtest result means something for paper.

## 2. Components: one engine, two drivers

The strategy, construction, risk and engine are shared. A backtest plugs in the simulated
broker; paper trading plugs in the order manager, which talks to Alpaca.

```mermaid
flowchart TB
    subgraph inputs["Inputs"]
        yaml["config/*.yaml<br/>presets and fixed settings"]
        env[".env<br/>Alpaca paper keys"]
        api[("Alpaca data API")]
    end

    cfg["config.py<br/>resolved recipe"]
    keys["keys.py<br/>redacted keys"]
    data["data.py<br/>bar cache, holdout cutoff<br/>HoldoutAccess / PaperAccess"]

    yaml --> cfg
    env --> keys
    api --> data
    keys --> data

    subgraph shared["Shared by backtest and paper"]
        view["view.py<br/>MarketView, past bars only"]
        strat["strategies/*.py<br/>on_bar returns Signals"]
        guard["guard.py<br/>blocks files, network"]
        cons["construction.py<br/>rank and size"]
        costs["costs.py"]
        risk["risk.py<br/>hard limits"]
        engine["engine.py<br/>daily loop, fill at next open"]
        brokerif["broker/base.py<br/>Broker interface"]
    end

    data --> view --> strat
    guard -. "wraps" .- strat
    strat -- "Signals" --> cons
    costs --> cons
    cons -- "Orders" --> risk
    risk -- "approved Orders" --> brokerif
    engine -- "drives" --> view
    engine -- "drives" --> strat
    cfg --> engine

    subgraph backtest["Backtest"]
        run["run.py<br/>CLI: harness.run"]
        sim["broker/sim.py<br/>simulated fills"]
        report["report.py<br/>metrics vs SPY"]
        bt[("backtests/<br/>run folders, index.csv")]
    end

    run --> engine
    brokerif -. "implemented by" .-> sim
    run --> report --> bt

    subgraph paper["Paper trading"]
        profiles["profiles.py<br/>promote, freeze, activate"]
        runner["paper.py<br/>CLI: harness.paper"]
        orders["orders.py<br/>order manager"]
        journal["journal.py<br/>append-only log"]
        positions["positions.py<br/>ledger, reconcile"]
        venue["broker/venue.py<br/>Venue protocol"]
        alpaca["broker/alpaca.py"]
        preport["paper_report.py"]
        kill["killswitch.py"]
        alerts["alerts.py"]
        inbox["inbox.py"]
        state[("state/paper/")]
    end

    bt -- "promote" --> profiles
    profiles -- "frozen profile" --> runner
    runner --> engine
    brokerif -. "implemented by" .-> orders
    orders --> venue -. "implemented by" .-> alpaca
    alpaca --> acct[("Alpaca paper account")]
    orders -- "write-ahead" --> journal
    journal --> positions
    positions -- "compare, never correct" --> alpaca
    journal --> preport
    kill -- "blocks sends" --> orders
    inbox -- "approve / reject" --> orders
    runner --> alerts
    journal --> state
    preport --> state
    alerts --> state

    subgraph operator["Operator"]
        cli["cli.py<br/>CLI: harness.cli"]
        service["service.py<br/>read-only views"]
        control["control.py<br/>commands"]
    end

    cli --> service
    cli --> control
    service -- "reads files only" --> state
    control --> kill
    control --> inbox
    control --> journal
```

## 3. Data models

The shapes that cross module boundaries (`harness/models.py`). Every order is a list of
legs, so a stock trade is one leg and an option spread is several.

```mermaid
classDiagram
    direction LR

    class Bar {
        symbol
        date
        open, high, low, close
        volume
    }

    class Instrument {
        <<union>>
    }
    class Stock {
        symbol
    }
    class OptionContract {
        underlying
        expiry
        strike
        right
        multiplier
    }
    Instrument <|-- Stock
    Instrument <|-- OptionContract

    class Leg {
        instrument
        side
        ratio
    }
    Leg --> Instrument

    class Signal {
        action
        legs
        strength
        reason
    }
    Signal "1" *-- "1..*" Leg

    class Order {
        created_on
        action
        legs
        quantity
        strength
        reason
    }
    Order "1" *-- "1..*" Leg
    Signal ..> Order : construction sizes

    class Fill {
        order_id
        signal_date, fill_date
        instrument
        side
        quantity
        reference_price, price
        slippage_cost, commission
    }
    Order "1" <.. "0..*" Fill : order_id

    class OrderUpdate {
        order_id
        status
        fills
        reason
    }
    OrderUpdate "1" *-- "0..*" Fill

    class Position {
        instrument
        quantity
        avg_price
        entry_date
        entry_order_id
    }
    Fill ..> Position : opens or closes

    class AccountState {
        cash
        positions
    }
    class PortfolioSnapshot {
        as_of
        cash
        equity
        positions
    }
    AccountState "1" *-- "0..*" Position
    PortfolioSnapshot "1" *-- "0..*" Position
```

The strategy sees `Bar`s (through `MarketView`) and a `PortfolioSnapshot`, and returns
`Signal`s. Everything to the right of `Signal` belongs to the harness.

## 4. Strategy lifecycle

```mermaid
flowchart LR
    write["Write strategy<br/>strategies/name.py<br/>+ preset"] --> bt["Backtest<br/>harness.run"]
    bt -- "iterate" --> write
    bt --> ho["Holdout run<br/>harness.run --holdout<br/>logged"]
    ho --> promote["Promote<br/>harness.paper promote<br/>frozen profile"]
    promote --> activate["Activate<br/>bind flat paper account"]
    activate --> daily["Paper run each morning<br/>harness.paper run"]
    daily -- "change anything" --> bt
```
