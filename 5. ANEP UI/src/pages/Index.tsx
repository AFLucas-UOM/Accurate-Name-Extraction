import NavBar from "@/components/layout/NavBar";
import Footer from "@/components/layout/Footer";
import VideoAnalyzer from "@/components/analyzer/VideoAnalyzer";
const Index = () => {
  return <div className="flex flex-col min-h-screen">
      <NavBar />
      
      <main className="flex-1 flex flex-col">
        <div className="bg-gradient-to-b from-blue-50 to-white dark:from-gray-900 dark:to-gray-800 pt-14 pb-6 my-[28px]\\n mb-[35px]">
          <div className="container px-4 text-center">
            <h1 className="text-3xl md:text-4xl lg:text-5xl font-bold mb-6 bg-clip-text text-transparent bg-gradient-to-r from-blue-600 to-indigo-600 dark:from-blue-400 dark:to-indigo-400">
              Accurate Name Extraction from News Video Graphics
            </h1>
            <p className="text-base text-gray-600 dark:text-gray-300 max-w-3xl mx-auto mb-8">
              Extract names from news video graphics with precision using advanced computer vision models
            </p>
          </div>
        </div>  
        
        <div className="flex-1 -mt-8 mb-[35px]">
          <VideoAnalyzer />
        </div>
      </main>
      <Footer />
    </div>;
};
export default Index;