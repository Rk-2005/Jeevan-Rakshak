import React, { useState, useEffect } from "react";
import { MapContainer, Marker, Popup, TileLayer, Polyline, useMapEvents } from "react-leaflet";
import MarkerClusterGroup from "react-leaflet-cluster";
import "./App.css";
import { Icon, divIcon } from "leaflet";
import "leaflet/dist/leaflet.css";
import { uploadJSONToFirestore , uploadJSONDynamically , uploadPolylinesToFirestore ,fetchNodeDataFromFirestore } 
from './FireApp'; // Import the function

const customIcon = new Icon({
  iconUrl: "https://cdn-icons-png.flaticon.com/512/5968/5968526.png",
  iconSize: [20, 20],
});

const createClusterCustomIcon = function (cluster) {
  return new divIcon({
    html: `<span class="cluster-icon">${cluster.getChildCount()}</span>`,
    className: "custom-marker-cluster",
    iconSize: [33, 33],
  });
};


export default function App() {
  const [userCoordinates, setUserCoordinates] = useState({
    latitude: "",
    longitude: "",
    popUp: "",
    IsContaminated : 0,
    IsLeaking : 0,
    CaseOfProliferation : 0,
  });

  function MyComponent() {
    const map = useMapEvents({
      click(e) {
        console.log("component used");
        const { lat, lng } = e.latlng;
        console.log(`Coordinates: Lat ${lat.toFixed(6)}, Lng ${lng.toFixed(6)}`);
        setUserCoordinates({
          latitude: lat.toFixed(6),
          longitude: lng.toFixed(6),
          popUp: "", // Clear the popUp field when clicking on the map
        });
        setCurrentPolylineStart({
          latitude: lat.toFixed(6),
          longitude: lng.toFixed(6),
        });
      },
    });
  
    return null; // Issue 1: Missing return statement
  }
  

  const [markers, setMarkers] = useState([]);
  useEffect(() => {
    // Fetch node data from Firestore
    const fetchNodeData = async () => {
      try {
        const nodeData = await fetchNodeDataFromFirestore();
        setMarkers(nodeData);
        console.log("It worked here");
      } catch (error) {
        console.error('Error fetching node data:', error);
      }
    };
    fetchNodeData();
  }, []);


  
  const [userPolylines, setUserPolylines] = useState([]);
  const [currentPolylineStart, setCurrentPolylineStart] = useState({
    latitude: "",
    longitude: "",
  });
  const [currentPolylineEnd, setCurrentPolylineEnd] = useState({
    latitude: "",
    longitude: "",
  });

  const handleAddMarker = () => {
    if (
      (userCoordinates.latitude && userCoordinates.longitude) &&
      userCoordinates.popUp
    ) {
      const newMarker = {
        geocode: [
          parseFloat(userCoordinates.latitude),
          parseFloat(userCoordinates.longitude),
        ],

        popUp: userCoordinates.popUp,
      };
      const jsonDataForNode ={
        latitude: parseFloat(userCoordinates.latitude),
        longitude:parseFloat(userCoordinates.longitude),
        popUp: userCoordinates.popUp,
        IsContaminated : 0,
        IsLeaking : 0,
        CaseOfProliferation : 0,
      };
      uploadJSONDynamically(jsonDataForNode); // Replace 'unique_document_name' with a unique identifier for the document
    
  

      setMarkers((prevMarkers) => [...prevMarkers, newMarker]);

      setUserCoordinates({
        latitude: "",
        longitude: "",
        popUp: "",
        IsContaminated : 0,
        IsLeaking : 0,
        CaseOfProliferation :0 ,
      });
    } else {
      // Handle error or validation message when not all input fields are filled
      // You can display an error message or handle it in your preferred way.
    }
  };

  const handleAddPolyline = () => {
    if (
      currentPolylineStart.latitude &&
      currentPolylineStart.longitude &&
      currentPolylineEnd.latitude &&
      currentPolylineEnd.longitude
    ) {
      const newPolyline = [
        [
          parseFloat(currentPolylineStart.latitude),
          parseFloat(currentPolylineStart.longitude),
        ],
        [
          parseFloat(currentPolylineEnd.latitude),
          parseFloat(currentPolylineEnd.longitude),
        ],
      ];

      setUserPolylines((prevPolylines) => [...prevPolylines, newPolyline]);
      setCurrentPolylineStart({
        latitude: "",
        longitude: "",
      });
      setCurrentPolylineEnd({
        latitude: "",
        longitude: "",
      });
    } else {
      // Handle error or validation message when the inputs are missing
      // You can display an error message or handle it in your preferred way.
    }
  };

  const handleMapClick = (e) => {
    console.log("Using the function");
    const { lat, lng } = e.latlng;
    
    // Update the input fields with the clicked coordinates
    setUserCoordinates({
      latitude: lat.toFixed(6),
      longitude: lng.toFixed(6),
      popUp: "", // Clear the popUp field when clicking on the map
    });
    console.log(userCoordinates.latitude)
    console.log(userCoordinates.longitude)
  };


  const handleMarkerClick = (coordinates) => {
    console.log(`Clicked Marker: Lat ${coordinates[0]}, Lng ${coordinates[1]}`);
    setCurrentPolylineEnd({
        latitude:coordinates[0],
        longitude:coordinates[1]
    })
    // You can perform any desired actions with the coordinates here
  };


  const getCoordinatesAsJSON = () => {
    const coordinatesJSON = markers.map((marker) => ({
      latitude: marker.geocode[0],
      longitude: marker.geocode[1],
      popUp: marker.popUp,
      IsContaminated : 0,
      IsLeaking : 0,
      CaseOfProliferation : 0,
    }));

    // Call the upload function from FireApp.js
    uploadJSONToFirestore(coordinatesJSON);
  };

  const getPolylinesAsJSON = () => {
    const polylinesJSON = userPolylines.map((polyline) => ({
      coordinates: polyline.map((coord) => ({
        latitude: coord[0],
        longitude: coord[1],
      })),
    }));
  
    // Call the upload function from FireApp.js to upload the polylines to Firestore
    uploadPolylinesToFirestore(polylinesJSON);
  };

  return (
    <div>
      <br />
      <br />
      <br />

      {/* Input fields for latitude and longitude */}
      <input
        type="text"
        placeholder="Latitude"
        value={userCoordinates.latitude}
        onChange={(e) =>
          setUserCoordinates({ ...userCoordinates, latitude: e.target.value })
          
        }
        
      />
      <input
        type="text"
        placeholder="Longitude"
        value={userCoordinates.longitude}
        onChange={(e) =>
          setUserCoordinates({ ...userCoordinates, longitude: e.target.value })
        }
      />

      <input
        type="text"
        placeholder="Popup Content"
        value={userCoordinates.popUp}
        onChange={(e) =>
          setUserCoordinates({ ...userCoordinates, popUp: e.target.value })
        }
      />
      <button onClick={handleAddMarker}>Add Marker</button>
      <button onClick={() => console.log(getPolylinesAsJSON())}>GetCoordinates</button>

      <hr />


      <input
        type="text"
        placeholder="Start Node Latitude"
        value={currentPolylineStart.latitude}
        onChange={(e) =>
          setCurrentPolylineStart({
            ...currentPolylineStart,
            latitude: e.target.value,
          })
        }
      />
      <input
        type="text"
        placeholder="Start Node Longitude"
        value={currentPolylineStart.longitude}
        onChange={(e) =>
          setCurrentPolylineStart({
            ...currentPolylineStart,
            longitude: e.target.value,
          })
        }
      />
      <input
        type="text"
        placeholder="End Node Latitude"
        value={currentPolylineEnd.latitude}
        onChange={(e) =>
          setCurrentPolylineEnd({
            ...currentPolylineEnd,
            latitude: e.target.value,
          })
        }
      />
      <input
        type="text"
        placeholder="End Node Longitude"
        value={currentPolylineEnd.longitude}
        onChange={(e) =>
          setCurrentPolylineEnd({
            ...currentPolylineEnd,
            longitude: e.target.value,
          })
        }
      />
      <button onClick={handleAddPolyline}>Add Polyline</button>

      <MapContainer center={[21.069082, 79.065665]} zoom={17.5}>
  <MyComponent /> {/* Move MyComponent inside the MapContainer */}
  {userPolylines.map((polyline, index) => (
    <Polyline key={index} positions={polyline} color="blue" />
  ))}
  <TileLayer
    attribution='&copy; OpenStreetMap'
    url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
  />
  <MarkerClusterGroup chunkedLoading iconCreateFunction={createClusterCustomIcon}>
    {markers.map((marker, index) => (
      <Marker key={index} position={marker.geocode} icon={customIcon} eventHandlers={{
        click: () => handleMarkerClick(marker.geocode),
      }}>
        <Popup>
          {marker.popUp}
        </Popup>
      </Marker>
    ))}
  </MarkerClusterGroup>
  </MyComponent>
</MapContainer>

      
    </div>
  );
}
