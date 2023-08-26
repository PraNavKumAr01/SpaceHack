import React, {useRef} from 'react';
import { OrbitControls, Sphere, Stars } from '@react-three/drei';
import PlanetMap from '../assets/Terrestrial1.png'
import NullMap from '../assets/Icy.png'
import NonHabitablePlanetMap from '../assets/Volcanic.png'
import NormalMap from '../assets/8k_earth_normal_map.jpg'
import SpecularMap from '../assets/8k_earth_specular_map.jpg'
import CloudMap from '../assets/8k_earth_clouds.jpg'
import { useLoader, useFrame } from '@react-three/fiber';
import { TextureLoader } from 'three';
import * as THREE from "three";

const Planet = ({ predictionData }) => {

    
    const colorMap = useLoader(TextureLoader, PlanetMap);
    const nullMap = useLoader(TextureLoader, NullMap);
    const nonHabitablePlanetMap = useLoader(TextureLoader, NonHabitablePlanetMap);
    const normalMap = useLoader(TextureLoader, NormalMap);
    const specularMap = useLoader(TextureLoader, SpecularMap);
    const cloudMap = useLoader(TextureLoader, CloudMap);

    const earthRef = useRef();
    const cloudsRef = useRef();

    useFrame(({ clock }) => {
        const elapsedTime = clock.getElapsedTime();
    
        earthRef.current.rotation.y = elapsedTime / 6;
        cloudsRef.current.rotation.y = elapsedTime / 4;

      });
      let habitability = predictionData;
      console.log(habitability)

  return (
    <>
    <pointLight color="#f6f3ea" position={[10, 10, 5]} intensity={4} />
    // <ambientLight intensity={2} />
    
    <mesh ref={cloudsRef} position={[-3.2, 0, 0]}>
        <sphereGeometry args={[2.005, 128, 128]} />
        <meshPhongMaterial
          map={cloudMap}
          opacity={habitability === true ? 0.5 :
            habitability === false ? 0 :
            0.5}
          depthWrite={true}
          transparent={true}
          side={THREE.DoubleSide}
        />
    </mesh>

    <mesh ref={earthRef} position={[-3.2, 0, 0]}>
        <sphereGeometry args={[2, 128, 128]} />
        <meshPhongMaterial specularMap={specularMap} />
        <meshStandardMaterial
          map={
            habitability === true ? colorMap : 
             predictionData === false ?  nonHabitablePlanetMap :
             nullMap
          }
          normalMap={normalMap}
          metalness={0.4}
          roughness={0.7}
        />
         <OrbitControls
          enableZoom={false}
          enablePan={true}
          enableRotate={true}
          zoomSpeed={0.6}
          panSpeed={0.5}
          rotateSpeed={0.4}
        /> 
      </mesh>

    </>
  );
};

export default Planet;