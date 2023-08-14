import { Outlet, NavLink, Link } from "react-router-dom";

import styles from "./Layout.module.css";

const Layout = () => {
    return (
        <div className={styles.layout}>
            <header className={styles.header} role={"banner"}>
                <div className={styles.headerContainer}>
                    <div>
                        <img src="/favicon.ico" width={100} className={styles.ImageContainer} />
                    </div>
                    <div>
                        <h5 className={styles.chatInputEngine}>Powered by Azure OpenAI</h5>
                    </div>
                </div>
            </header>

            <Outlet />
        </div>
    );
};

export default Layout;
