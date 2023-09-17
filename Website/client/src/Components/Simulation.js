import React from 'react'
import Request from '../Tools/Request';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  LineElement,
  PointElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';
import { Line } from 'react-chartjs-2';

ChartJS.register(
  CategoryScale,
  LinearScale,
  LineElement,
  PointElement,
  Title,
  Tooltip,
  Legend
);

const ChartOptions = {
  responsive: true,
  interaction: {
    mode: 'index',
    intersect: false,
  },
  stacked: false,
  plugins: {
    title: {
      display: true,
      text: 'Trading Activity',
    },
  },
  scales: {
    y: {
      type: 'linear',
      display: true,
      position: 'left',
    },
    y1: {
      type: 'linear',
      display: true,
      position: 'right',
      grid: {
        drawOnChartArea: false,
      },
    },
  },
};

class Simulation extends React.Component {
    constructor(props) {
      super(props)
      this.state={Statistics: null};
    }

    componentDidMount() {
      Request.CacheableRequest('GET','model/'+this.props.stock+'/run').then(Result=>{
        this.setState({Statistics:Result.Body});
      });
    }

    render() {
      if(!this.state.Statistics)
        return <></>;
      return <div className="bg-dark p-4 text-center" style={{width:'60%', border: '1px solid white'}}>
        <h1>Simulation Results:</h1>
        <p>Profit from holding: ${this.state.Statistics.HoldingProfit.toFixed(2)}</p>
        <p>Profit from trading: ${this.state.Statistics.TradingProfit.toFixed(2)}</p>
        <p>Percent gains from holding: {this.state.Statistics.HoldingPercent.toFixed(2)}%</p>
        <p>Percent gains from trading: {this.state.Statistics.TradingPercent.toFixed(2)}%</p>
        <p>Gains vs. market: {this.state.Statistics.PercentDifference.toFixed(2)}%</p>
        <Line options={ChartOptions} data={{labels:this.state.Statistics.DatesRaw,datasets: [
          {
            label: 'Share Price',
            data: this.state.Statistics.SharePriceRaw.map(Datum => Datum.toFixed(2)),
            borderColor: 'rgba(255, 99, 132,1)', backgroundColor: 'rgba(255, 99, 132, 1)',yAxisID: 'y',
          },
          {
            label: 'Net Worth',
            data: this.state.Statistics.NetWorth.map(Datum => Datum.toFixed(2)),
            borderColor: 'rgb(53, 162, 235)',
            backgroundColor: 'rgba(53, 162, 235, 1)',
            yAxisID: 'y1',
          },
          {
            label: 'Trading Volume',
            data: this.state.Statistics.TradingVolume.map(Datum => Datum.toFixed(2)),
            borderColor: 'rgb(53, 235, 162)',
            backgroundColor: 'rgba(53, 235, 162, 1)',
            yAxisID: 'y1',
          },
          // {
          //   label: 'Equity Percent',
          //   data: this.state.Statistics.EquityPercent.map(Datum => 100*Datum.toFixed(2)),
          //   borderColor: 'rgb(53, 235, 162)',
          //   backgroundColor: 'rgba(53, 235, 162, 1)',
          //   yAxisID: 'y',
          // },
        ]}} />
      </div>
    }
}

export default Simulation;
