import { useState } from "react";
import "./App.css";
import PlayerSimilarity from "./PlayerSimilarity";
import PlayerGraph from "./PlayerGraph";

function App() {
  const [c_df2, setC_df2] = useState<{
    LR1: number[];
    LR2: number[];
    Cluster: number[];
    Name: string[];
  } | null>(null);
  const [df_with_names, setDf_with_names] = useState<any[]>([]);

  return (
    <div className="App">
      <h1>NBA Player Similarity Project</h1>
      <PlayerGraph
        c_df2={c_df2}
        setC_df2={setC_df2}
        df_with_names={df_with_names}
        setDf_with_names={setDf_with_names}
      />
      <PlayerSimilarity c_df2={c_df2} df_with_names={df_with_names} />
    </div>
  );
}

export default App;
