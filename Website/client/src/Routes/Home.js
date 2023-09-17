import React from 'react'
import Button from 'react-bootstrap/Button';
import NavBar from '../Components/NavBar';
import Simulation from '../Components/Simulation';


class Home extends React.Component {

  textValue = '';
  
  constructor(props) {
    super(props);
    this.state={stock: ''};
    this.runSimulation=this.runSimulation.bind(this);
    this.updateText=this.updateText.bind(this);
  }

  runSimulation() {
    this.setState({stock: this.textValue});
  }
  updateText(Event) {
    this.textValue = Event.target.value;
  }

  render() {
    return (
      <div>
        <NavBar/>
        <div className="text-center py-3">
          <h1>Proof of Concen: Simulation</h1>
          <p style={{margin: 'auto',width:'60%'}}>
            Simply input a stock ticker, and our neural network will train on that stock's data up to the end of 2014. It will subsequently forecast the stock's values from 2015 to 2019. An algorithm will then execute daily stock buying and selling decisions based on these predictions. The resulting transaction data will be presented in a graph below for your exploration and enjoyment. Give it a try!
          </p>
          <div className="d-flex justify-content-around">
            <div className="p-4 d-flex align-items-center">
              <input type="text" className="me-4 p-2 ps-3" onChange={this.updateText} placeholder="Enter Stock" style={{border: '1px solid white', color: 'black', borderRadius: '5em'}}/>
              <Button variant="success" onClick={this.runSimulation}>Run Simulation</Button>
            </div>
          </div>
          {this.state.stock.length?(
            <div className="d-flex justify-content-around">
              <Simulation key={this.state.stock} stock={this.state.stock}/>
            </div>
          ):<></>}
        </div>
      </div>
    );
  }
}

export default Home;