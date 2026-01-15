import React, { useEffect, useState } from "react";

const ComplainsData = () => {
  const [complaints, setComplaints] = useState([]);

  useEffect(() => {
    // Dummy complaint data for PS 25001
    const dummyData = [
      {
        id: "C001",
        date: "2025-09-10",
        description:
          "Several villagers reported diarrhea cases after consuming water from the community tube well.",
        imageUrl: "https://via.placeholder.com/150",
        location: "Village A, Assam",
        type: "Health Alert",
        priority: "High",
      },
      {
        id: "C002",
        date: "2025-09-12",
        description:
          "Water sample from hand pump shows high turbidity and bacterial presence.",
        imageUrl: "",
        location: "Village B, Meghalaya",
        type: "Water Quality",
        priority: "Medium",
      },
      {
        id: "C003",
        date: "2025-09-14",
        description:
          "Typhoid suspected in 3 children. Urgent medical intervention needed.",
        imageUrl: "https://via.placeholder.com/150",
        location: "Remote Hamlet, Nagaland",
        type: "Health Alert",
        priority: "High",
      },
      {
        id: "C004",
        date: "2025-09-15",
        description:
          "Community reporting foul smell and color change in drinking water storage tank.",
        imageUrl: "",
        location: "Block C, Manipur",
        type: "Water Quality",
        priority: "Low",
      },
      {
        id: "C005",
        date: "2025-09-15",
        description:
          "ASHA worker reports rise in fever and stomach infection cases after monsoon floods.",
        imageUrl: "https://via.placeholder.com/150",
        location: "Village D, Tripura",
        type: "Health Alert",
        priority: "Medium",
      },
    ];

    setComplaints(dummyData);
  }, []);

  // Priority badge with colors
  const getPriorityBadge = (priority) => {
    let bgColor = "";
    switch (priority) {
      case "High":
        bgColor = "bg-red-500 text-white";
        break;
      case "Medium":
        bgColor = "bg-yellow-400 text-black";
        break;
      case "Low":
        bgColor = "bg-green-500 text-white";
        break;
      default:
        bgColor = "bg-gray-300 text-black";
    }
    return (
      <span
        className={`px-3 py-1 rounded-full text-sm font-semibold ${bgColor}`}
      >
        {priority}
      </span>
    );
  };

  return (
    <div className="m-2 md:m-10 mt-24 p-2 md:p-10 bg-white rounded-3xl shadow-lg">
      <h2 className="text-2xl font-bold mb-6 text-gray-700">
        Community Health & Water Complaints
      </h2>
      <div className="overflow-x-auto">
        <table className="min-w-full bg-white rounded-lg overflow-hidden shadow-md">
          <thead className="bg-gray-200 text-gray-700 text-sm uppercase font-semibold">
            <tr>
              <th className="py-3 px-4 text-left">ID</th>
              <th className="py-3 px-4 text-left">Date</th>
              <th className="py-3 px-4 text-left">Description</th>
              <th className="py-3 px-4 text-left">Image</th>
              <th className="py-3 px-4 text-left">Location</th>
              <th className="py-3 px-4 text-left">Type</th>
              <th className="py-3 px-4 text-center">Priority</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-200">
            {complaints.map((complaint) => (
              <tr key={complaint.id} className="hover:bg-gray-50">
                <td className="py-3 px-4">{complaint.id}</td>
                <td className="py-3 px-4">{complaint.date}</td>
                <td className="py-3 px-4">{complaint.description}</td>
                <td className="py-3 px-4">
                  {complaint.imageUrl ? (
                    <button
                      type="button"
                      onClick={() => window.open(complaint.imageUrl, "_blank")}
                    >
                      <img
                        src={complaint.imageUrl}
                        alt="Complaint"
                        className="w-20 h-auto rounded-lg shadow"
                        onError={(e) => {
                          e.target.style.display = "none";
                        }}
                      />
                    </button>
                  ) : (
                    <span className="text-gray-400 italic">No image</span>
                  )}
                </td>
                <td className="py-3 px-4">{complaint.zone} - {complaint.address}</td>
                <td className="py-3 px-4">{complaint.type}</td>
                <td className="py-3 px-4 text-center">
                  {getPriorityBadge(complaint.priority)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default ComplainsData;
