import { TooltipComponent } from '@syncfusion/ej2-react-popups';
import React, { useState } from 'react';
import { Link, NavLink, useNavigate } from 'react-router-dom';
import { AiOutlineShoppingCart } from 'react-icons/ai';
import { MdOutlineCancel } from 'react-icons/md';
import logo from '../data/jeevanrakshak-logo.png';
import { useStateContext } from '../contexts/ContextProvider';
import { links } from '../data/dummy';
import { MapPin } from "lucide-react";
import { useLanguage } from '../contexts/LanguageContext';

const Sidebar = () => {
  const { currentColor, activeMenu, setActiveMenu, screenSize } = useStateContext();
  const { t } = useLanguage();

  const handleCloseSideBar = () => {
    if (activeMenu !== undefined && screenSize <= 900) {
      setActiveMenu(false);
    }
  };

  const [is,setis]=useState(false);
  const activeLink = 'flex items-center gap-4 pl-4 pr-3 py-2.5 rounded-xl text-white text-[15px] m-2 shadow-sm';
  const normalLink = 'flex items-center gap-4 pl-4 pr-3 py-2.5 rounded-xl text-[15px] text-gray-700 dark:text-gray-200 dark:hover:text-black hover:bg-light-gray m-2 transition-colors';
  const navigate = useNavigate();
  
  // Get user role from localStorage
  const userRole = localStorage.getItem('userRole');
  const userId = localStorage.getItem('userId');
  
  // Filter links based on user role
  const filterLinks = (links) => {
    if (!userId) {
      // Not logged in - show only login
      return links.filter(section => section.title === 'Administration').map(section => ({
        ...section,
        links: section.links.filter(link => link.name === 'login')
      }));
    }
    
    if (userRole === 'admin') {
      // Admin - show everything EXCEPT Community section
      return links.filter(section => {
        if (section.title === 'Community') {
          return false; // Hide entire Community section for admin
        }
        if (section.title === 'Administration') {
          return true;
        }
        return true; // Show Dashboard, Health Monitoring, GIS-MAPPING, Apps, etc.
      }).map(section => {
        if (section.title === 'Administration') {
          return {
            ...section,
            links: section.links.filter(link => link.name === 'complaint-management')
          };
        }
        // Filter Dashboard to show only admin dashboard
        if (section.title === 'Dashboard') {
          return {
            ...section,
            links: section.links.filter(link => link.role === 'admin' || !link.role)
          };
        }
        // Filter Health Monitoring to show only admin links
        if (section.title === 'Health Monitoring') {
          return {
            ...section,
            links: section.links.filter(link => link.role === 'admin' || !link.role)
          };
        }
        return section;
      });
    } else if (userRole === 'asha-worker') {
      // ASHA Worker - show ONLY Dashboard and Health Monitoring sections (NO GIS-MAPPING)
      return links.filter(section => 
        (section.title === 'Dashboard' || section.title === 'Health Monitoring') && 
        section.title !== 'GIS-MAPPING'
      ).map(section => {
        if (section.title === 'Dashboard') {
          return {
            ...section,
            links: section.links.filter(link => link.role === 'asha-worker')
          };
        }
        return {
          ...section,
          links: section.links.filter(link => link.role === 'asha-worker')
        };
      });
    } else {
      // Regular user - show ONLY Dashboard and Community sections (NO GIS-MAPPING)
      return links.filter(section => 
        (section.title === 'Dashboard' || section.title === 'Community') && 
        section.title !== 'GIS-MAPPING'
      ).map(section => {
        if (section.title === 'Dashboard') {
          return {
            ...section,
            links: section.links.filter(link => link.role === 'user')
          };
        }
        return {
          ...section,
          links: section.links.filter(link => 
            link.name !== 'view-complaints' && link.name !== 'complaint-management'
          )
        };
      });
    }
  };
  
  const filteredLinks = filterLinks(links);
  return (
    <div className="ml-3 h-screen md:overflow-hidden overflow-auto md:hover:overflow-auto pb-10">
      {activeMenu && (
        <>
          <div className="flex justify-between items-center">
            <Link to="/" onClick={handleCloseSideBar} className="items-center gap-3 ml-10 mt-4 flex text-xl font-extrabold tracking-tight dark:text-white text-slate-900">
              <img onClick={()=>{
                navigate("/")
              }}src={logo} alt="JeevanRakshak" className="h-12 w-13" />
            </Link>
            <TooltipComponent content="Menu" position="BottomCenter">
              <button
                type="button"
                aria-label="Close sidebar"
                onClick={() => setActiveMenu(!activeMenu)}
                style={{ color: currentColor }}
                className="text-xl rounded-full p-3 hover:bg-light-gray mt-4 block md:hidden"
              >
                <MdOutlineCancel />
              </button>
            </TooltipComponent>
          </div>
          
          <div className="mt-8">
            {filteredLinks.map((item) => {
              // Translate section titles
              const getTranslatedTitle = (title) => {
                switch(title) {
                  case 'Dashboard':
                    return t('dashboardMenu');
                  case 'Health Monitoring':
                    return t('healthMonitoringMenu');
                  case 'GIS-MAPPING':
                    return t('gisMapping');
                  case 'Administration':
                    return t('administrationMenu');
                  case 'Apps':
                    return t('appsMenu');
                  case 'Community':
                    return t('community') || 'Community';
                  default:
                    return title;
                }
              };

              return (
              <div key={item.title}>
             
                <p className="text-gray-400 dark:text-gray-400 m-3 mt-4 uppercase">
                  {getTranslatedTitle(item.title)}
                </p>
                {item.links.map((link) => (
                  <NavLink
                    to={`/${link.name}`}
                    key={link.name}
                    onClick={handleCloseSideBar}
                    style={({ isActive }) => ({
                      backgroundColor: isActive ? currentColor : '',
                    })}
                    className={({ isActive }) => (isActive ? activeLink : normalLink)}
                  >
                    {link.icon}
                    <span className="capitalize ">{link.name}</span>
                  </NavLink>
                ))}
              
              {item.title === "Dashboard" && userRole === 'admin' &&  
  <div key="GISMapping">
    <p className="text-gray-400 dark:text-gray-400 m-3 mt-4 uppercase">
      {t('gisMapping')}
    </p>
    <a
      href="https://health-rakshak.vercel.app/"
      onClick={handleCloseSideBar}
      className={window.location.pathname.includes('gisTracking') ? activeLink : normalLink}
    >
      {/* Icon on the left */}
      <MapPin className="w-5 h-5" /> 

      {/* Label */}
      <span className="capitalize font-medium">gisTracking</span>
    </a>
  </div>
}

              </div>
            );
            })}
            
           
          </div>
        </>
      )}
    </div>
  );
};

export default Sidebar;
