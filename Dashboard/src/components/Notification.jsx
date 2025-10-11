import React, { useState } from 'react';
import { MdOutlineCancel } from 'react-icons/md';
import { FaCheck } from 'react-icons/fa';

import Button from './Button';
import { chatData as initialChatData } from '../data/dummy';
import { useStateContext } from '../contexts/ContextProvider';

const Notification = () => {
  const { currentColor } = useStateContext();

  // State to manage notifications
  const [notifications, setNotifications] = useState(initialChatData);

  // Function to mark a notification as done (remove it)
  const handleDone = (index) => {
    setNotifications((prev) => prev.filter((_, i) => i !== index));
  };

  const getDotColor = (priority) => {
    switch (priority) {
      case 'high':
        return 'bg-red-500';
      case 'medium':
        return 'bg-orange-400';
      case 'low':
        return 'bg-green-500';
      default:
        return 'bg-gray-400';
    }
  };

  return (
    <div className="nav-item absolute right-5 md:right-40 top-16 bg-white dark:bg-[#42464D] p-8 rounded-lg w-96 shadow-lg">
      {/* Header */}
      <div className="flex justify-between items-center border-b border-gray-200 dark:border-gray-600 pb-3">
        <div className="flex gap-3 items-center">
          <p className="font-semibold text-lg dark:text-gray-200">Notifications</p>
          <button
            type="button"
            className="text-white text-xs rounded p-1 px-2 bg-orange-theme"
          >
            {notifications.length} New
          </button>
        </div>
        <Button
          icon={<MdOutlineCancel />}
          color="rgb(153, 171, 180)"
          bgHoverColor="light-gray"
          size="2xl"
          borderRadius="50%"
        />
      </div>

      {/* Notifications List */}
      <div className="mt-5 max-h-72 overflow-y-auto custom-scrollbar">
        {notifications.length > 0 ? (
          notifications.map((item, index) => (
            <div
              key={index}
              className="flex items-center justify-between gap-3 p-4 border-b border-gray-200 dark:border-gray-700 hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors duration-200 rounded-lg"
            >
              {/* Left side: Dot + Content */}
              <div className="flex items-start gap-3 flex-1">
                {/* Colored Dot */}
                <span
                  className={`w-3 h-3 mt-2 rounded-full ${getDotColor(
                    item.priority
                  )}`}
                ></span>

                {/* Message + Description */}
                <div className="flex-1">
                  <div className="flex justify-between items-center">
                    <p className="font-semibold text-gray-800 dark:text-gray-200">
                      {item.message}
                    </p>
                    <span className="text-xs text-gray-400">{item.time}</span>
                  </div>
                  <p className="text-sm text-gray-600 dark:text-gray-400">
                    {item.desc}
                  </p>
                </div>
              </div>

              {/* Right side: Done (✔) button */}
              <button
                className="text-green-500 hover:text-green-700 p-2 rounded-full hover:bg-green-100 dark:hover:bg-green-900/30"
                onClick={() => handleDone(index)}
              >
                <FaCheck />
              </button>
            </div>
          ))
        ) : (
          <p className="text-center text-gray-500 dark:text-gray-400 py-6">
            No new notifications 🎉
          </p>
        )}
      </div>

      {/* Footer */}
      {notifications.length > 0 && (
        <div className="mt-5">
          <Button
            color="white"
            bgColor={currentColor}
            text="See all notifications"
            borderRadius="10px"
            width="full"
          />
        </div>
      )}
    </div>
  );
};

export default Notification;
