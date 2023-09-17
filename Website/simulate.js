function Simulate(Data, StartingBalance, StartingInvestment, SensitivityBuy, SensitivitySell, MaxTrade) {
    let Balance = StartingBalance;
    let Shares = 0;
    let Stats = { TradingVolume: [], NetWorth: [], EquityPercent: []};
    let Trade = (Amount, Price) => {
        //We can't sell more than we have
        if(Amount < 0 && Math.abs(Amount) > Shares)
            Amount = -Shares;
        //We can't buy more than we can afford
        if(Amount > 0 && Amount * Price > Balance)
            Amount = Math.floor(Balance / Price);

        Shares += Amount;
        Balance -= Price * Amount;
        return Price * Amount;
    }

    //Buy our initial shares
    Trade((StartingInvestment*Balance)/Data[0].Actual, Data[0].Actual);

    //Simulate each day
    for(let Day=0; Day<Data.length-1; Day++) {
        //Trade the stonks
        let PredictedDelta = Data[Day+1].Predicted - Data[Day].Actual;
        let PredictedPercent = PredictedDelta/Data[Day].Actual*100;
        Stats.TradingVolume[Day] = Trade(PredictedPercent*(PredictedPercent>0?SensitivityBuy:SensitivitySell)/Data[Day].Actual, Data[Day].Actual);
        Stats.NetWorth[Day] = Balance + Shares * Data[Day].Actual;
        Stats.EquityPercent[Day] = 1 - Balance / Stats.NetWorth[Day];
    }

    //Sell the stonks
    Trade(-Shares, Data[Data.length-1].Actual);

    //Calculate Statistics
    Stats.TradingProfit = Balance - StartingBalance;
    Stats.TradingPercent = Stats.TradingProfit / StartingBalance * 100;
    Stats.HoldingProfit = Data[Data.length-1].Actual / Data[0].Actual * StartingBalance - StartingBalance;
    Stats.HoldingPercent = Stats.HoldingProfit / StartingBalance * 100;
    Stats.PercentDifference = ((Stats.TradingPercent + 100) / (Stats.HoldingPercent + 100) - 1) * 100;
    return Stats;
}

module.exports = Simulate;
