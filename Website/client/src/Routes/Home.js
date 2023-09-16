import React from 'react'

import NavBar from '../Components/NavBar';

class Home extends React.Component {
  
  constructor(props) {
    super(props);
  }

  render() {
    return (
      <div>
        <NavBar/>
        <div>
          <h1>Available Models:</h1>
        </div>
      </div>
    );
  }
}



export default Home;