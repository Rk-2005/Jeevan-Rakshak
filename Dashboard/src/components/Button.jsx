import { useStateContext } from '../contexts/ContextProvider';

const Button = ({ icon, bgColor, color, bgHoverColor, size, text, borderRadius, width, onClick, disabled, className, type = "button" }) => {
  const { setIsClicked, initialState } = useStateContext();

  // Map size to actual Tailwind classes
  const getSizeClass = (size) => {
    switch (size) {
      case 'sm': return 'text-sm';
      case 'md': return 'text-base';
      case 'lg': return 'text-lg';
      case 'xl': return 'text-xl';
      default: return 'text-base';
    }
  };

  // Map width to actual Tailwind classes
  const getWidthClass = (width) => {
    switch (width) {
      case 'sm': return 'w-20';
      case 'md': return 'w-32';
      case 'lg': return 'w-40';
      case 'xl': return 'w-48';
      case 'full': return 'w-full';
      default: return 'w-auto';
    }
  };

  return (
    <button
      type={type}
      onClick={onClick || (() => setIsClicked(initialState))}
      style={{ backgroundColor: bgColor, color, borderRadius }}
      className={`${getSizeClass(size)} p-3 ${getWidthClass(width)} hover:drop-shadow-xl ${className || ''}`}
      disabled={disabled}
    >
      {icon && <span className="mr-2">{icon}</span>}
      {text}
    </button>
  );
};

export default Button;
