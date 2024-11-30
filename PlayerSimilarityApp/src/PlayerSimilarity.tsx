import { useEffect, useState } from "react";
import axios from "axios";
import "./styles.css";

interface PlayerSimilarityProps {
  c_df2: {
    LR1: number[];
    LR2: number[];
    Cluster: number[];
    Name: string[];
  } | null;
  df_with_names: any[];
}
export default function PlayerSimilarity(props: PlayerSimilarityProps) {
  const [playerName, setPlayerName] = useState("");
  const [numberPlayers, setNumberPlayers] = useState(5);
  const [postResponse, setPostResponse] = useState<string[] | null>(null);

  useEffect(() => {
    setPostResponse(null);
  }, []);

  const handleButtonClick = async () => {
    try {
      const response = await axios.post("http://127.0.0.1:5000/predict", {
        playerName,
        numberPlayers,
        c_df2: props.c_df2,
        df_with_names: props.df_with_names,
      });
      setPostResponse(response.data);
    } catch (e) {
      console.log(e);
    }
  };
  const handleNumberChange = (value: string) => {
    const re = /^[0-9\b]+$/;
    if (value === "" || re.test(value)) {
      setNumberPlayers(Number(value));
    }
  };
  console.log(postResponse);

  return (
    props.c_df2 != null && (
      <div className="player-similarity">
        <p className="player-similarity-text">
          Nice! Above is a 2d visual representation of the clusters we created!
          Different clusters are grouped by color, and you can hover your mouse
          over a data point to view the player name. You can also query our
          results to get the closest n players to a specific player. Below, set
          your desired player, and the number of closest players to that person!
        </p>
        <div className="find-bar">
          <p>Find </p>
          <input
            defaultValue={5}
            value={numberPlayers}
            onChange={(e) => handleNumberChange(e.target.value)}
            className="find-number"
          />
          <p> similar players to</p>

          <input
            placeholder="player name"
            onChange={(e) => setPlayerName(e.target.value)}
            className="find-player"
          />
          <button className="find-button" onClick={handleButtonClick}>
            go
          </button>
        </div>
        {postResponse !== null && postResponse.length === 0 ? (
          <p style={{ textAlign: "center" }}>Player not found</p>
        ) : postResponse !== null && postResponse.length > 0 ? (
          postResponse.map((player) => (
            <p style={{ textAlign: "center" }} key={player}>
              {player}
            </p>
          ))
        ) : null}
      </div>
    )
  );
}
