import React from 'react'

import { UserContext } from '../Contexts/User';
import Button from 'react-bootstrap/Button';

import './NavBar.css';

class Home extends React.Component {
  
  constructor(props) {
    super(props);
  }

  render() {
    return (
        <div className="d-flex justify-content-between align-items-center tape-bg" style={{height: '5em',borderBottom: '1px solid white'}}>
            <div className="d-flex align-items-center tape-cover-left">
                <img className="mx-2" src="/logo.png" style={{height: '3.5em'}}/>
                <p className="m-0" style={{fontSize: '2em'}}>WolframWallstreet</p>
            </div>
            <div className="d-flex align-items-center tape-cover-right">
                <p className="m-0" style={{fontSize: '1em'}}>Welcome, {this.context.User.nickname}</p>
                <Button className="mx-3" href="/logout" variant="danger">Logout</Button>
            </div>
        </div>
    );
  }
}


Home.contextType = UserContext;

export default Home;