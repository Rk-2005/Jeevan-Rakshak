import React, { useState } from "react";
import { MapContainer, Marker, Popup, TileLayer, Polyline, useMapEvents } from "react-leaflet";
import MarkerClusterGroup from "react-leaflet-cluster";
import "./App.css";
import { Icon, divIcon } from "leaflet";
import "leaflet/dist/leaflet.css";

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

        })
      },
    });
  
    return null;
  }
  

  const [markers, setMarkers] = useState([
    {
      geocode: [21.1490, 79.0890],
      popUp: "Hello, I am pop up 1"
    },
    {
      geocode: [21.1467, 79.0867],
      popUp: "Hello, I am pop up 2"
    },
    {
      geocode: [21.1501, 79.0801],
      popUp: "Hello, I am pop up 3"
    },
    {
      geocode: [21.071285,79.065530],
      popUp: "Crown Society Water Tank"
    },
    {
      geocode: [21.070747,79.065341],
      popUp: "Node 1"
    },
    {
      geocode: [21.070308,79.065248],
      popUp: "Node 2"
    },
    {
      geocode: [21.070211,79.065853],
      popUp: "Node 3"
    },
    {
      geocode: [21.069082,79.065665],
      popUp: "Node 4"
    },
    {
      geocode: [21.069256,79.064727],
      popUp: "Node 5"
    },
    {
      geocode: [21.068633,79.065626],
      popUp: "Node 6"
    },
    {
      geocode: [21.068837,79.064682],
      popUp: "Node 7"
    },
    {
      geocode: [21.068407,79.064586],
      popUp: "Node 8"
    },
    {
      geocode: [21.070102,79.066756],
      popUp: "Node 9"
    },
    {
      geocode: [21.070545,79.066820],
      popUp: "Node 10"
    },
    {
      geocode : [21.070015,79.067354],
      popUp: "Node 11"
    },
    {
      geocode: [21.069682,79.066709],
      popUp: "Node 12"
    },
    {
      geocode: [21.069764,79.066254],
      popUp: "Node 13"
    },
    {
      geocode: [21.067962,79.066011],
      popUp: "Node 14"
    },
    {
      geocode: [21.071395,79.067521],
      popUp: "Node 15"
    },
    {
      geocode: [21.069860,79.068613],
      popUp: "Node 16"
    }
  ]);
  const [userPolylines, setUserPolylines] = useState([
    [
        [21.1490, 79.0890], // Start coordinate
        [21.1467, 79.0867], // End coordinate
      ],
      [
        [21.071285,79.065530], //Tank
        [21.070747,79.065341],  //Node1
      ],
      [
        [21.070747,79.065341],  //Node1
        [21.070308,79.065248],  //Node2
      ],
      [
        [21.070308,79.065248],  //Node2
        [21.070211,79.065853],  //Node3
      ],
      [
        [21.070211,79.065853],  //Node3
        [21.069082,79.065665],  //Node4
      ],
      [
        [21.069082,79.065665],  //Node4
        [21.069256,79.064727],  //Node5
      ],
      [
        [21.069082,79.065665],  //Node4
        [21.068633,79.065626],  //Node6
      ],
      [
        [21.069256,79.064727],  //Node5
        [21.068837,79.064682],  //Node7
      ],
      [
        [21.068837,79.064682],  //Node7
        [21.068407,79.064586],  //Node8
      ],
      [
        [21.070211,79.065853],  //Node3
        [21.070102,79.066756],  //Node9 
      ],
      [
        [21.070102,79.066756],  //Node9 
        [21.070545,79.066820],  //Node10
      ],
      [
        [21.070015,79.067354],  //Node11
        [21.070102,79.066756],  //Node9 
      ],
      [
        [21.069682,79.066709],  //Node12
        [21.070102,79.066756],  //Node9 
      ],
      [
        [21.069764,79.066254],  //Node13
        [21.069682,79.066709],  //Node12
      ],
      [
        [21.069764,79.066254],  //Node13
        [21.067962,79.066011],  //Node14
      ],
      [
        [21.070015,79.067354],  //Node11
        [21.071395,79.067521],  //Node15
      ],
      [
        [21.070015,79.067354],  //Node11
        [21.069860,79.068613],  //Node16
      ],
  ]);
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

      setMarkers((prevMarkers) => [...prevMarkers, newMarker]);

      setUserCoordinates({
        latitude: "",
        longitude: "",
        popUp: "",
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

  const handleSetLatitude = () => {
    setUserCoordinates({
      ...userCoordinates,
      latitude: "42.123456",
    });
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
    const coordinatesJSON = markers.map((marker, index) => ({
      id: index + 1,
      latitude: marker.geocode[0],
      longitude: marker.geocode[1],
      popUp: marker.popUp,
    }));

    return JSON.stringify(coordinatesJSON, null, 2);
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
      <button onClick={() => console.log(getCoordinatesAsJSON())}>GetCoordinates</button>

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
        <Popup>{marker.popUp}</Popup>
      </Marker>
    ))}
  </MarkerClusterGroup>
</MapContainer>
      
    </div>
  );
}
