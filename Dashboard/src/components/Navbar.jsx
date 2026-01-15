import { TooltipComponent } from '@syncfusion/ej2-react-popups';
import React, { useEffect } from 'react';
import { AiOutlineMenu } from 'react-icons/ai';
import { MdKeyboardArrowDown } from 'react-icons/md';
import { RiNotification3Line } from 'react-icons/ri';

import Cart from './Cart';
import Chat from './Chat';
import Notification from './Notification';   // ✅ water alerts dropdown
import UserProfile from './UserProfile';
import LanguageSwitcher from './LanguageSwitcher';
import { useStateContext } from '../contexts/ContextProvider';
import avatar from '../data/JR3.png';

const NavButton = ({ title, customFunc, icon, color, dotColor }) => (
  <TooltipComponent content={title} position="BottomCenter">
    <button
      type="button"
      onClick={() => customFunc()}
      style={{ color }}
      className="relative text-xl rounded-full p-3 hover:bg-light-gray"
    >
      {dotColor && (
        <span
          style={{ background: dotColor }}
          className="absolute inline-flex rounded-full h-2 w-2 right-2 top-2"
        />
      )}
      {icon}
    </button>
  </TooltipComponent>
);

const Navbar = () => {
  const {
    currentColor,
    activeMenu,
    setActiveMenu,
    handleClick,
    isClicked,
    setScreenSize,
    screenSize,
  } = useStateContext();

  useEffect(() => {
    const handleResize = () => setScreenSize(window.innerWidth);

    window.addEventListener('resize', handleResize);
    handleResize();

    return () => window.removeEventListener('resize', handleResize);
  }, []);

  useEffect(() => {
    if (screenSize <= 900) {
      setActiveMenu(false);
    } else {
      setActiveMenu(true);
    }
  }, [screenSize]);

  const handleActiveMenu = () => setActiveMenu(!activeMenu);
  
  const userName = localStorage.getItem('userName') || 'User';
  const userZone = localStorage.getItem('userZone') || '';

  return (
    <div className="flex justify-between p-2 md:ml-6 md:mr-6 relative border-b border-gray-100 dark:border-gray-700/50">

      {/* Left Side - Menu button */}
      <NavButton
        title="Menu"
        customFunc={handleActiveMenu}
        color={currentColor}
        icon={<AiOutlineMenu />}
      />

      {/* Right Side */}
      <div className="flex items-center gap-2">
        <LanguageSwitcher />

        {/* ✅ Notification Icon (Water Alerts) */}
        <NavButton
          title="Notification"
          dotColor="rgb(254, 201, 15)"
          customFunc={() => handleClick('notification')}
          color={currentColor}
          icon={<RiNotification3Line />}
        />

        {/* Profile */}
        <TooltipComponent content="Profile" position="BottomCenter">
          <div
            className="flex items-center gap-2 cursor-pointer p-1 hover:bg-light-gray rounded-lg transition-colors"
            onClick={() => handleClick('userProfile')}
          >
            <div className="hidden md:flex flex-col items-end mr-2">
              <p className="text-sm font-semibold text-gray-700 dark:text-gray-200">{userName}</p>
              {userZone && <p className="text-xs text-gray-500 dark:text-gray-400">{userZone}</p>}
            </div>
            <img
              className="avatar rounded-full w-10 h-10"
              src={avatar}
              alt="user-profile"
            />
            <MdKeyboardArrowDown className="text-gray-400 text-14" />
          </div>
        </TooltipComponent>

        {/* Popups */}
        {isClicked.cart && <Cart />}
        {isClicked.chat && <Chat />}
        {isClicked.notification && <Notification />}  {/* ✅ water-related dropdown */}
        {isClicked.userProfile && <UserProfile />}
      </div>
    </div>
  );
};

export default Navbar;
