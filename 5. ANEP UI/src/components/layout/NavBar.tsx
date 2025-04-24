import { Moon, Sun } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useTheme } from "@/hooks/use-theme";

const NavBar = () => {
  const { theme, toggleTheme } = useTheme();

  return (
    <header className="border-b px-6 py-3 flex items-center justify-between">
      <a
        href="/"
        className="ml-2 flex items-center space-x-2 transform transition-transform duration-300 ease-in-out hover:scale-105 focus:outline-none"
      >
        <h1 className="text-xl font-bold text-anep-blue dark:text-blue-400">ANEP</h1>
        <span className="text-xs bg-anep-lightblue text-white px-2 py-0.5 rounded">Beta</span>
      </a>

      <Button
        variant="ghost"
        size="icon"
        onClick={toggleTheme}
        className="rounded-full"
      >
        {theme === "dark" ? <Sun className="h-5 w-5" /> : <Moon className="h-5 w-5" />}
      </Button>
    </header>
  );
};

export default NavBar;
