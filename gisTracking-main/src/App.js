import React, { useState } from "react";
import { MapContainer, Marker, Popup, TileLayer, Polyline, useMapEvents } from "react-leaflet";
import MarkerClusterGroup from "react-leaflet-cluster";
import "./App.css";
import { Icon, divIcon } from "leaflet";
import "leaflet/dist/leaflet.css";
import { uploadJSONToFirestore , uploadJSONDynamically , uploadPolylinesToFirestore } from './FireApp'; // Import the function

const customIcon = new Icon({
  iconUrl: "https://cdn-icons-png.flaticon.com/512/5968/5968526.png",
  iconSize: [20, 20],
});

const createClusterCustomIcon = function (cluster) {
  return new divIcon({
    html: `<span class="cluster-icon">${cluster.getChildCount()}</span>`,
    className: "custom-marker-cluster",
    iconSize: [50, 50],
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
      geocode: [21.071285,79.065530],
      popUp: "Source1"
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
    {
      positions: [[21.1490, 79.0890],[21.1467, 79.0867]],
      meta: { from: 'Node X', to: 'Node Y', risk: 57.5, cases: 5, weather: 'Light rainfall', priority: 'Medium', waterQuality: 'pH slightly unsafe', action: 'Monitor water source' }
    },
    {
      positions: [[21.071285,79.065530],[21.070747,79.065341]],
      meta: { from: 'Tank', to: 'Node 1', risk: 42.0, cases: 2, weather: 'Clear', priority: 'Low', waterQuality: 'Safe', action: 'No action' }
    },
    {
      positions: [[21.070747,79.065341],[21.070308,79.065248]],
      meta: { from: 'Node 1', to: 'Node 2', risk: 57.5, cases: 5, weather: 'Light rainfall', priority: 'Medium', waterQuality: 'pH slightly unsafe', action: 'Monitor water source' }
    },
    {
      positions: [[21.070308,79.065248],[21.070211,79.065853]],
      meta: { from: 'Node 2', to: 'Node 3', risk: 71.0, cases: 7, weather: 'Humid', priority: 'High', waterQuality: 'Turbidity high', action: 'Dispatch field team' }
    },
    {
      positions: [[21.070211,79.065853],[21.069082,79.065665]],
      meta: { from: 'Node 3', to: 'Node 4', risk: 34.0, cases: 1, weather: 'Clear', priority: 'Low', waterQuality: 'Safe', action: 'No action' }
    },
    {
      positions: [[21.069082,79.065665],[21.069256,79.064727]],
      meta: { from: 'Node 4', to: 'Node 5', risk: 63.0, cases: 3, weather: 'Overcast', priority: 'Medium', waterQuality: 'Chlorine low', action: 'Increase chlorination' }
    },
    {
      positions: [[21.069082,79.065665],[21.068633,79.065626]],
      meta: { from: 'Node 4', to: 'Node 6', risk: 28.0, cases: 0, weather: 'Clear', priority: 'Low', waterQuality: 'Safe', action: 'No action' }
    },
    {
      positions: [[21.069256,79.064727],[21.068837,79.064682]],
      meta: { from: 'Node 5', to: 'Node 7', risk: 48.0, cases: 2, weather: 'Light rainfall', priority: 'Medium', waterQuality: 'pH slightly unsafe', action: 'Monitor water source' }
    },
    {
      positions: [[21.068837,79.064682],[21.068407,79.064586]],
      meta: { from: 'Node 7', to: 'Node 8', risk: 22.0, cases: 0, weather: 'Clear', priority: 'Low', waterQuality: 'Safe', action: 'No action' }
    },
    {
      positions: [[21.070211,79.065853],[21.070102,79.066756]],
      meta: { from: 'Node 3', to: 'Node 9', risk: 69.0, cases: 6, weather: 'Stormy', priority: 'High', waterQuality: 'Turbidity high', action: 'Issue boil-water advisory' }
    },
    {
      positions: [[21.070102,79.066756],[21.070545,79.066820]],
      meta: { from: 'Node 9', to: 'Node 10', risk: 40.0, cases: 1, weather: 'Humid', priority: 'Medium', waterQuality: 'Chlorine low', action: 'Increase chlorination' }
    },
    {
      positions: [[21.070015,79.067354],[21.070102,79.066756]],
      meta: { from: 'Node 11', to: 'Node 9', risk: 31.0, cases: 1, weather: 'Clear', priority: 'Low', waterQuality: 'Safe', action: 'No action' }
    },
    {
      positions: [[21.069682,79.066709],[21.070102,79.066756]],
      meta: { from: 'Node 12', to: 'Node 9', risk: 55.0, cases: 3, weather: 'Light rainfall', priority: 'Medium', waterQuality: 'pH slightly unsafe', action: 'Monitor water source' }
    },
    {
      positions: [[21.069764,79.066254],[21.069682,79.066709]],
      meta: { from: 'Node 13', to: 'Node 12', risk: 37.0, cases: 1, weather: 'Overcast', priority: 'Low', waterQuality: 'Safe', action: 'No action' }
    },
    {
      positions: [[21.069764,79.066254],[21.067962,79.066011]],
      meta: { from: 'Node 13', to: 'Node 14', risk: 45.0, cases: 2, weather: 'Humid', priority: 'Medium', waterQuality: 'Chlorine low', action: 'Increase chlorination' }
    },
    {
      positions: [[21.070015,79.067354],[21.071395,79.067521]],
      meta: { from: 'Node 11', to: 'Node 15', risk: 76.0, cases: 8, weather: 'Stormy', priority: 'High', waterQuality: 'Turbidity high', action: 'Dispatch field team' }
    },
    {
      positions: [[21.070015,79.067354],[21.069860,79.068613]],
      meta: { from: 'Node 11', to: 'Node 16', risk: 33.0, cases: 1, weather: 'Clear', priority: 'Low', waterQuality: 'Safe', action: 'No action' }
    },
  ]);

  const [selectedEdge, setSelectedEdge] = useState(null);
  const [showControls, setShowControls] = useState(false);
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
      const newPolyline = {
        positions: [
          [parseFloat(currentPolylineStart.latitude), parseFloat(currentPolylineStart.longitude)],
          [parseFloat(currentPolylineEnd.latitude), parseFloat(currentPolylineEnd.longitude)],
        ],
        meta: { from: 'Node A', to: 'Node B', risk: 25, cases: 0, weather: 'Clear', priority: 'Low', waterQuality: 'Safe', action: 'No action' }
      };

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
    const polylinesJSON = userPolylines.map((edge) => ({
      coordinates: edge.positions.map((coord) => ({
        latitude: coord[0],
        longitude: coord[1],
      })),
      meta: edge.meta,
    }));
  
    // Call the upload function from FireApp.js to upload the polylines to Firestore
    uploadPolylinesToFirestore(polylinesJSON);
  };

  return (
    <div>
      {/* Floating toggle for controls */}
      <button
        onClick={() => setShowControls((v) => !v)}
        style={{ position: 'fixed', top: 12, left: 12, zIndex: 2100, background: '#2563eb', color: '#ffffff', border: 'none', padding: '8px 10px', borderRadius: '8px', cursor: 'pointer' }}
      >
        {showControls ? 'Hide Controls' : 'Show Controls'}
      </button>

      {/* Controls overlay */}
      {showControls && (
        <div style={{ position: 'fixed', top: 52, left: 12, zIndex: 2050, background: '#ffffff', border: '1px solid rgba(0,0,0,0.1)', boxShadow: '0 8px 24px rgba(0,0,0,0.08)', borderRadius: '10px', padding: '10px', width: '320px', maxHeight: '80vh', overflowY: 'auto' }}>
          <div style={{ fontWeight: 700, marginBottom: '8px' }}>Map Controls</div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '6px' }}>
            <input type="text" placeholder="Latitude" value={userCoordinates.latitude} onChange={(e) => setUserCoordinates({ ...userCoordinates, latitude: e.target.value })} />
            <input type="text" placeholder="Longitude" value={userCoordinates.longitude} onChange={(e) => setUserCoordinates({ ...userCoordinates, longitude: e.target.value })} />
          </div>
          <input type="text" placeholder="Popup Content" value={userCoordinates.popUp} onChange={(e) => setUserCoordinates({ ...userCoordinates, popUp: e.target.value })} style={{ marginTop: '6px', width: '100%' }} />
          <div style={{ display: 'flex', gap: '8px', marginTop: '8px' }}>
            <button onClick={handleAddMarker}>Add Marker</button>
            <button onClick={() => console.log(getPolylinesAsJSON())}>GetCoordinates</button>
          </div>

          <hr style={{ margin: '10px 0' }} />

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '6px' }}>
            <input type="text" placeholder="Start Node Latitude" value={currentPolylineStart.latitude} onChange={(e) => setCurrentPolylineStart({ ...currentPolylineStart, latitude: e.target.value })} />
            <input type="text" placeholder="Start Node Longitude" value={currentPolylineStart.longitude} onChange={(e) => setCurrentPolylineStart({ ...currentPolylineStart, longitude: e.target.value })} />
            <input type="text" placeholder="End Node Latitude" value={currentPolylineEnd.latitude} onChange={(e) => setCurrentPolylineEnd({ ...currentPolylineEnd, latitude: e.target.value })} />
            <input type="text" placeholder="End Node Longitude" value={currentPolylineEnd.longitude} onChange={(e) => setCurrentPolylineEnd({ ...currentPolylineEnd, longitude: e.target.value })} />
          </div>
          <button onClick={handleAddPolyline} style={{ marginTop: '8px' }}>Add Polyline</button>
        </div>
      )}

      <MapContainer center={[21.069082, 79.065665]} zoom={17.5} style={{ position: 'fixed', inset: 0, right: selectedEdge ? 360 : 0, height: '100vh', width: 'auto' }}>
  <MyComponent /> {/* Move MyComponent inside the MapContainer */}
  {userPolylines.map((edge, index) => (
    <Polyline
      key={index}
      positions={edge.positions}
      pathOptions={{ color: edge.meta?.priority === 'High' ? 'red' : edge.meta?.priority === 'Medium' ? 'orange' : 'green', weight: 5, opacity: 0.9 }}
      eventHandlers={{
        click: () => setSelectedEdge(edge),
      }}
    />
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

      {/* Details Side Panel */}
      {selectedEdge && (
        <div style={{ position: 'fixed', top: 0, right: 0, height: '100vh', width: '360px', background: '#ffffff', borderLeft: '1px solid rgba(0,0,0,0.1)', boxShadow: '-8px 0 24px rgba(0,0,0,0.06)', padding: '16px', overflowY: 'auto', zIndex: 2000 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
            <h3 style={{ fontSize: '18px', fontWeight: 700 }}>Outbreak Details</h3>
            <button onClick={() => setSelectedEdge(null)} style={{ border: 'none', background: 'transparent', fontSize: '20px', cursor: 'pointer' }} aria-label="Close panel">×</button>
          </div>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '14px' }}>
            <tbody>
              <tr>
                <td style={{ border: '1px solid #e5e7eb', padding: '8px', width: '55%', background: '#f8fafc' }}>From – To Location</td>
                <td style={{ border: '1px solid #e5e7eb', padding: '8px' }}>{selectedEdge.meta.from}  →  {selectedEdge.meta.to}</td>
              </tr>
              <tr>
                <td style={{ border: '1px solid #e5e7eb', padding: '8px', background: '#f8fafc' }}>Average Outbreak Risk (%)</td>
                <td style={{ border: '1px solid #e5e7eb', padding: '8px' }}>{selectedEdge.meta.risk}%</td>
              </tr>
              <tr>
                <td style={{ border: '1px solid #e5e7eb', padding: '8px', background: '#f8fafc' }}>Recent Reported Cases</td>
                <td style={{ border: '1px solid #e5e7eb', padding: '8px' }}>{selectedEdge.meta.cases}</td>
              </tr>
              <tr>
                <td style={{ border: '1px solid #e5e7eb', padding: '8px', background: '#f8fafc' }}>Weather Condition</td>
                <td style={{ border: '1px solid #e5e7eb', padding: '8px' }}>{selectedEdge.meta.weather}</td>
              </tr>
              <tr>
                <td style={{ border: '1px solid #e5e7eb', padding: '8px', background: '#f8fafc' }}>Priority Level</td>
                <td style={{ border: '1px solid #e5e7eb', padding: '8px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <span style={{ display: 'inline-block', width: '10px', height: '10px', borderRadius: '9999px', background: selectedEdge.meta.priority === 'High' ? '#ef4444' : selectedEdge.meta.priority === 'Medium' ? '#f59e0b' : '#10b981' }} />
                  {selectedEdge.meta.priority}
                </td>
              </tr>
              <tr>
                <td style={{ border: '1px solid #e5e7eb', padding: '8px', background: '#f8fafc' }}>Water Quality</td>
                <td style={{ border: '1px solid #e5e7eb', padding: '8px' }}>{selectedEdge.meta.waterQuality}</td>
              </tr>
              <tr>
                <td style={{ border: '1px solid #e5e7eb', padding: '8px', background: '#f8fafc' }}>Recommended Action</td>
                <td style={{ border: '1px solid #e5e7eb', padding: '8px' }}>{selectedEdge.meta.action}</td>
              </tr>
            </tbody>
          </table>
        </div>
      )}
      
    </div>
  );
}
