import { Canvas } from "@react-three/fiber";
import {default as Planet} from "./Planet/Planet";
import { styled } from "styled-components";
import { Suspense, useState } from "react";
import { Stars } from "@react-three/drei";
import {default as Form} from './Form/Form'

const CanvasContainer = styled.div`
  width: 100%;
  height: 100%;
  background-color: #000;
  margin: 0;
  padding: 0;
`

const Prediction = styled.h1`
  position: absolute;
  width: 100%;
  height: 10%;
  top: 85%;
  left: 17%;
  color: #fff;
  font-size: 50px;
  pointer-events: none;
  z-index: 1000;
`

function App() {
  const [predictionData, setPredictionData] = useState(null);

  const updatePredictionData = (data) => {
    setPredictionData(data);
};

console.log(predictionData)

  return (
    <CanvasContainer>
    <Canvas >
    <Suspense fallback={null} >
    <Stars
    radius={100}
    depth={60}
    count={10000}
    factor={7}
    saturation={0}
    fade={true}
    />
    <Planet predictionData={predictionData} />
    </Suspense>
    </Canvas>

    <Form onDataReceived={updatePredictionData} />
    <Prediction>
    {
    predictionData === true ? "Habitable" :
    predictionData === false ? "Non-Habitable" 
    : "Let's Find Out"
    }
    </Prediction>

    </CanvasContainer>
  );
}

export default App;
