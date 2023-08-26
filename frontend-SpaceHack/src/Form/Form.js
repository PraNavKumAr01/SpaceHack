import React from "react";
import styled from "styled-components";
import { useState } from 'react';
import axios, { all } from 'axios'
import './style.css';


const TopSectionContainer = styled.div`
  position: absolute;
  width: 50%;
  height: 100%;
  top: 10%;
  left: 47%;
  padding-top: 0%;
  border: 1px solid white
  display: flex;
  justify-content: center;
  align-items: center;
`;




export default function Form({ onDataReceived }) {

    const [OrbitalPeriod,setOrbitalPeriod] = useState("");
    const [PlanetaryRadius,setPlanetaryRadius] = useState("");
    const [EquilibriumTemperature,setEquilibriumTemperature] = useState("");
    const [InsolationFlux,setInsolationFlux] = useState("");
    const [StellarSurfaceGravity,setStellarSurfaceGravity] = useState("");

  
  
    const submitHandler = async (e) => {
        e.preventDefault();

        const { data } = await axios.post('https://spacehackdeploy-production-39bb.up.railway.app/predict', {OrbitalPeriod,PlanetaryRadius,EquilibriumTemperature,InsolationFlux,StellarSurfaceGravity},
            {
                headers: {
                    "Content-Type": "application/json",
                },
                withCredentials: true,
            } 
        );

        onDataReceived(data);
        console.log(data)
    }


  return (
    <TopSectionContainer>
            <div className="mainContainer">
                <form onSubmit={submitHandler}>

                    <div className="card">
                        <input
                            type="text"
                            name="OrbitalPeriod"
                            placeholder="Orbital Period"

                            value={OrbitalPeriod}
                            onChange={(e) => setOrbitalPeriod(e.target.value)}
                        />
                        {/* <label>Orbital Period</label> */}
                    </div>
                    <div className="card">
                        <input
                            type="text"
                            name="PlanetaryRadius"
                            placeholder="Planetary Radius (in Earth's Radius)"
                            value={PlanetaryRadius}
                            onChange={(e) => setPlanetaryRadius(e.target.value)}
                        />
                        {/* <label>Planetary Radius </label> */}
                    </div>
                    <div className="card">
                        <input
                            type="text"
                            name="EquilibriumTemperature"
                            placeholder="Equilibrium Temperature (K)"
                            value={EquilibriumTemperature}
                            onChange={(e) => setEquilibriumTemperature(e.target.value)}
                        />
                        {/* <label>Equilibrium Temperature</label> */}
                    </div>
                    <div className="card">
                        <input
                            type="text"
                            name="InsolationFlux"
                            placeholder="Magnetic Isolated Flux"
                            value={InsolationFlux}
                            onChange={(e) => setInsolationFlux(e.target.value)}
                        />
                        {/* <label>Magnetic Isolated Flux</label> */}
                    </div>
                    <div className="card">
                        <input
                            type="text"
                            name="StellarSurfaceGravity"
                            placeholder="Stellar Surface Gravity"
                            value={StellarSurfaceGravity}
                            onChange={(e) => setStellarSurfaceGravity(e.target.value)}
                        />
                        {/* <label>Surface Gravity</label> */}
                    </div>
                    <button type='submit' href="/" className='btn'>
                        <span></span>
                        <span></span>
                        <span></span>
                        <span></span>
                        Submit
                    </button>

                </form>
            </div>


    </TopSectionContainer>
  );
}