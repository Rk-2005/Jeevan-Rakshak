import React from 'react';

const Header = ({ category, title }) => (
  <div className=" mb-10">
    <p className="text-sm tracking-wide uppercase text-gray-400">{category}</p>
    <h2 className="text-2xl font-semibold mt-1 text-slate-900 dark:text-gray-100">{title}</h2>
  </div>
);

export default Header;
