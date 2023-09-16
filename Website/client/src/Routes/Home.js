import React from 'react'

import { UserContext } from '../Contexts/User';

class Home extends React.Component {
  
  constructor(props) {
    super(props);
    this.state = {};
  }

  render() {
    return (
            <p>Let the trading begin, {this.context.User.nickname}</p>
    );
  }
}


Home.contextType = UserContext;

export default Home;